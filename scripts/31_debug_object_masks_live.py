import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import robosuite as suite

from src.living_room.mask_grasp_planner import (
    extract_mask_features,
    select_object_feature,
    shape_hint,
    draw_features,
)


def make_env(args):
    kwargs = dict(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=args.cameras,
        camera_heights=args.height,
        camera_widths=args.width,
        camera_depths=True,
        camera_segmentations="element",
        reward_shaping=True,
        control_freq=20,
        horizon=500,
        single_object_mode=2,
        object_type=args.object_type,
    )

    return suite.make(**kwargs)


def get_img(obs, cam):
    return obs.get(f"{cam}_image", None)


def get_seg(obs, cam):
    keys = [
        f"{cam}_segmentation_element",
        f"{cam}_segmentation_instance",
        f"{cam}_segmentation_class",
    ]
    for k in keys:
        if k in obs:
            return obs[k], k
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object_type", default="milk", choices=["milk", "bread", "can", "cereal"])
    p.add_argument("--object_name", default=None)
    p.add_argument("--cameras", nargs="+", default=["agentview", "frontview"])
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--save_dir", default="outputs/mask_debug")
    p.add_argument("--no_show", action="store_true")
    args = p.parse_args()

    if args.object_name is None:
        args.object_name = {
            "milk": "mug",
            "bread": "book",
            "can": "remote",
            "cereal": "toy",
        }[args.object_type]

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("\n=== Mini-RT2-Room Mask Debug ===")
    print("object_type:", args.object_type)
    print("object_name:", args.object_name)
    print("cameras:", args.cameras)
    print("This checks segmentation masks, bbox, aspect ratio, yaw, and grasp hints.\n")

    env = make_env(args)
    obs = env.reset()

    low, high = env.action_spec
    action = np.zeros_like(low)

    try:
        for t in range(args.frames):
            obs, reward, done, info = env.step(action)
            env.render()

            panels = []

            for cam in args.cameras:
                img = get_img(obs, cam)
                seg, seg_key = get_seg(obs, cam)

                if img is None or seg is None:
                    if t == 0:
                        print("Missing image or seg for", cam)
                        print("Available obs keys:")
                        for k in sorted(obs.keys()):
                            print(" ", k)
                    continue

                img = np.asarray(img)
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)

                feats = extract_mask_features(seg, sim=env.sim, min_area=80)
                selected = select_object_feature(feats, args.object_name)
                vis = draw_features(img, feats[:10], selected)

                if selected:
                    shape, z_offset, use_yaw = shape_hint(args.object_name, selected)

                    txt1 = f"{cam}: selected={selected.geom_name[:35]}"
                    txt2 = f"bbox={selected.bbox_xywh} ar={selected.aspect_ratio:.2f} yaw={selected.major_axis_deg:.1f}"
                    txt3 = f"shape={shape} z={z_offset:.3f} use_yaw={use_yaw}"

                    cv2.putText(vis, txt1, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
                    cv2.putText(vis, txt2, (5, args.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                    cv2.putText(vis, txt3, (5, args.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

                panels.append(vis)

                if t % 30 == 0:
                    print(f"\n[t={t}] camera={cam} seg_key={seg_key} num_features={len(feats)}")
                    for i, f in enumerate(feats[:8]):
                        print(
                            f"  {i:02d} id={f.seg_id:<4} name={f.geom_name:<35} "
                            f"area={f.area_px:<6} bbox={f.bbox_xywh} "
                            f"aspect={f.aspect_ratio:.2f} yaw={f.major_axis_deg:.1f}"
                        )
                    if selected:
                        shape, z_offset, use_yaw = shape_hint(args.object_name, selected)
                        print("  SELECTED:", selected.geom_name)
                        print("  SHAPE:", shape, "Z_OFFSET:", z_offset, "USE_YAW:", use_yaw)

            if panels:
                panel = np.concatenate(panels, axis=1)

                if t % 60 == 0:
                    out = Path(args.save_dir) / f"{args.object_type}_{t:04d}.png"
                    cv2.imwrite(str(out), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

                if not args.no_show:
                    cv2.imshow("Mini-RT2-Room mask debug", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

            if done:
                obs = env.reset()

            time.sleep(0.02)

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
