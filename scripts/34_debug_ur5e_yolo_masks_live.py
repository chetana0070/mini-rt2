import argparse
import os
import time
import cv2
import numpy as np

from src.living_room.pickplace_env import make_pickplace_env
from src.living_room.ur5e_yolo_detector import YoloMaskDetector, draw_yolo_detections


def get_camera_image(obs, camera_name):
    key = f"{camera_name}_image"
    if key not in obs:
        image_keys = [k for k in obs.keys() if k.endswith("_image")]
        raise KeyError(f"{key} not found. Available image keys: {image_keys}")

    img = obs[key]
    img = np.asarray(img)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # robosuite images are often upside down depending on renderer.
    img = np.flipud(img)

    # Keep RGB internally.
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="UR5e", choices=["UR5e", "Panda"])
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ["ROBOT_NAME"] = args.robot

    print("\n=== UR5e YOLO Mask Live Debug ===")
    print("robot:", args.robot)
    print("camera:", args.camera)
    print("model:", args.model)
    print("conf:", args.conf)
    print("Press q to quit.\n")

    detector = YoloMaskDetector(model_path=args.model, conf=args.conf)

    env, camera_names = make_pickplace_env(
        render=True,
        robot=args.robot,
        camera_height=args.height,
        camera_width=args.width,
        horizon=args.horizon,
    )

    print("env cameras:", camera_names)

    obs = env.reset()

    writer = None
    if args.save_video:
        os.makedirs("videos/living_room", exist_ok=True)
        out_path = "videos/living_room/ur5e_yolo_masks_live.mp4"
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (args.width * 2, args.height),
        )
        print("video:", out_path)

    try:
        for t in range(args.horizon):
            action = np.zeros(env.action_dim, dtype=np.float32)
            obs, reward, done, info = env.step(action)

            env.render()

            img_rgb = get_camera_image(obs, args.camera)
            detections = detector.detect(img_rgb)

            overlay_rgb = draw_yolo_detections(img_rgb, detections)

            raw_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

            if detections:
                best = detections[0]
                print(
                    f"t={t:04d} best={best.cls_name} conf={best.conf:.2f} "
                    f"center=({best.center_px[0]:.1f},{best.center_px[1]:.1f}) "
                    f"bbox={best.bbox_xyxy} area={best.area_px} yaw={best.yaw_deg:.1f}"
                )
            elif t % 20 == 0:
                print(f"t={t:04d} no YOLO masks")

            panel = np.hstack([raw_bgr, overlay_bgr])

            cv2.putText(
                panel,
                "RAW CAMERA",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                "YOLO SEG MASKS",
                (args.width + 8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("UR5e YOLO mask debug", panel)

            if writer is not None:
                writer.write(panel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if done:
                obs = env.reset()

            time.sleep(0.01)

    finally:
        if writer is not None:
            writer.release()
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
