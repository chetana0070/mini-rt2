import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import robosuite as suite


CLASS_MAP = {
    "milk": 0,
    "can": 1,
    "bread": 2,
    "cereal": 3,
}

ALIASES = {
    "milk": ["milk"],
    "can": ["can"],
    "bread": ["bread"],
    "cereal": ["cereal"],
}


def seg_to_2d(seg):
    seg = np.asarray(seg)
    seg = np.squeeze(seg)
    if seg.ndim == 3:
        seg = seg[:, :, 0]
    return seg.astype(np.int32)


def get_geom_name(env, seg_id):
    try:
        return env.sim.model.geom_id2name(int(seg_id)) or f"seg_{seg_id}"
    except Exception:
        return f"seg_{seg_id}"


def object_mask_from_seg(env, seg, object_type):
    aliases = ALIASES[object_type]
    full_mask = np.zeros(seg.shape, dtype=np.uint8)

    ids = np.unique(seg)
    used = []

    for sid in ids:
        if sid < 0:
            continue

        name = get_geom_name(env, sid)
        low = name.lower()

        if any(a in low for a in aliases):
            m = (seg == sid).astype(np.uint8)
            if int(m.sum()) > 20:
                full_mask |= m
                used.append((int(sid), name, int(m.sum())))

    return full_mask, used


def mask_to_yolo_polygon(mask, width, height):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None

    eps = 0.006 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)

    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None

    # YOLO segmentation format:
    # class x1 y1 x2 y2 ... normalized
    poly = []
    for x, y in pts:
        poly.append(float(x) / float(width))
        poly.append(float(y) / float(height))

    return poly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="UR5e")
    parser.add_argument("--object_type", default="milk", choices=list(CLASS_MAP.keys()))
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--out", default="datasets/ur5e_yolo_seg")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    out = Path(args.out)
    img_dir = out / "images" / "train"
    lab_dir = out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting YOLO-seg labels")
    print("object_type:", args.object_type)
    print("camera:", args.camera)
    print("out:", out)

    env = suite.make(
        env_name="PickPlace",
        robots=args.robot,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[args.camera],
        camera_heights=args.height,
        camera_widths=args.width,
        camera_segmentations="element",
        single_object_mode=2,
        object_type=args.object_type,
        horizon=5,
    )

    saved = 0

    for i in range(args.n):
        obs = env.reset()

        # Let scene settle.
        for _ in range(2):
            action = np.zeros(env.action_dim, dtype=np.float32)
            obs, reward, done, info = env.step(action)

        img_key = f"{args.camera}_image"
        seg_key = f"{args.camera}_segmentation_element"

        if img_key not in obs or seg_key not in obs:
            print("Missing keys.")
            print("image keys:", [k for k in obs if k.endswith("_image")])
            print("seg keys:", [k for k in obs if "segmentation" in k])
            break

        img = np.asarray(obs[img_key])
        img = np.flipud(img)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        img = img.astype(np.uint8)

        seg = seg_to_2d(obs[seg_key])
        seg = np.flipud(seg)

        mask, used = object_mask_from_seg(env, seg, args.object_type)

        if int(mask.sum()) < 30:
            print(f"[{i}] no object mask. used={used}")
            continue

        poly = mask_to_yolo_polygon(mask, args.width, args.height)
        if poly is None:
            print(f"[{i}] bad polygon. mask_area={int(mask.sum())}")
            continue

        cls_id = CLASS_MAP[args.object_type]

        stem = f"{args.object_type}_{saved:05d}"
        img_path = img_dir / f"{stem}.jpg"
        lab_path = lab_dir / f"{stem}.txt"

        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        with open(lab_path, "w") as f:
            vals = " ".join(f"{v:.6f}" for v in poly)
            f.write(f"{cls_id} {vals}\n")

        if saved % 25 == 0:
            print(f"saved={saved} mask_area={int(mask.sum())} used={used[:3]}")

        saved += 1

    env.close()

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"""path: {out.resolve()}
train: images/train
val: images/train

names:
  0: milk
  1: can
  2: bread
  3: cereal
"""
    )

    print("Done.")
    print("saved:", saved)
    print("yaml:", yaml_path)


if __name__ == "__main__":
    main()
