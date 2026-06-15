import argparse
import os
import time
import cv2
import numpy as np
import robosuite as suite

from src.living_room.ur5e_yolo_detector import YoloMaskDetector, draw_yolo_detections


def get_camera_image(obs, camera_name):
    img = np.asarray(obs[f"{camera_name}_image"])

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    img = np.flipud(img)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="UR5e")
    parser.add_argument("--object_type", default="milk")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--model", default="runs/segment/train/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=2000)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ["ROBOT_NAME"] = args.robot

    print("\n=== Forced UR5e YOLO Milk Live ===")
    print("robot:", args.robot)
    print("object_type:", args.object_type)
    print("camera:", args.camera)
    print("model:", args.model)
    print("conf:", args.conf)
    print("Press q to quit.\n")

    detector = YoloMaskDetector(model_path=args.model, conf=args.conf)

    env = suite.make(
        env_name="PickPlace",
        robots=args.robot,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[args.camera],
        camera_heights=args.height,
        camera_widths=args.width,
        single_object_mode=2,
        object_type=args.object_type,
        horizon=args.horizon,
    )

    obs = env.reset()

    for t in range(args.horizon):
        action = np.zeros(env.action_dim, dtype=np.float32)
        obs, reward, done, info = env.step(action)

        env.render()

        img_rgb = get_camera_image(obs, args.camera)
        detections = detector.detect(img_rgb)

        overlay_rgb = draw_yolo_detections(img_rgb, detections)

        raw_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        panel = np.hstack([raw_bgr, overlay_bgr])

        if detections:
            best = detections[0]
            print(
                f"t={t:04d} best={best.cls_name} conf={best.conf:.2f} "
                f"center=({best.center_px[0]:.1f},{best.center_px[1]:.1f}) "
                f"bbox={best.bbox_xyxy} area={best.area_px} yaw={best.yaw_deg:.1f}"
            )
        elif t % 20 == 0:
            print(f"t={t:04d} no YOLO masks")

        cv2.imshow("FORCED milk YOLO mask debug", panel)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if done:
            obs = env.reset()

        time.sleep(0.01)

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
