import argparse
import os
import time
import cv2
import numpy as np
import robosuite as suite

from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
)

from src.living_room.ur5e_yolo_detector import YoloMaskDetector, draw_yolo_detections


def get_rgb(obs, camera):
    img = np.asarray(obs[f"{camera}_image"])
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.flipud(img)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def get_depth(obs, env, camera):
    key = f"{camera}_depth"
    depth = np.asarray(obs[key])
    depth = np.squeeze(depth)
    depth = np.flipud(depth)
    return get_real_depth_map(env.sim, depth)


def pixel_to_world(env, camera, u, v, depth, height, width):
    K = get_camera_intrinsic_matrix(env.sim, camera, height, width)
    E = get_camera_extrinsic_matrix(env.sim, camera)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    cam_point = np.array([x, y, z, 1.0], dtype=np.float64)
    world_point = E @ cam_point
    return world_point[:3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="UR5e")
    parser.add_argument("--object_type", default="milk")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--model", default="runs/segment/train/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=500)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ["ROBOT_NAME"] = args.robot

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
        camera_depths=True,
        single_object_mode=2,
        object_type=args.object_type,
        horizon=args.horizon,
    )

    obs = env.reset()

    print("\n=== YOLO mask → depth → world debug ===")
    print("Press q to quit.\n")

    for t in range(args.horizon):
        action = np.zeros(env.action_dim, dtype=np.float32)
        obs, _, done, _ = env.step(action)
        env.render()

        rgb = get_rgb(obs, args.camera)
        depth_map = get_depth(obs, env, args.camera)

        detections = detector.detect(rgb)
        overlay = draw_yolo_detections(rgb, detections)

        if detections:
            det = detections[0]
            u, v = det.center_px

            mask_depths = depth_map[det.mask > 0]
            mask_depths = mask_depths[np.isfinite(mask_depths)]
            mask_depths = mask_depths[mask_depths > 0]

            if len(mask_depths) > 0:
                d = float(np.median(mask_depths))
                world = pixel_to_world(env, args.camera, u, v, d, args.height, args.width)

                sim_obj = obs["Milk_pos"] if "Milk_pos" in obs else None

                print(
                    f"t={t:04d} cls={det.cls_name} conf={det.conf:.2f} "
                    f"px=({u:.1f},{v:.1f}) depth={d:.3f} "
                    f"world=({world[0]:.3f},{world[1]:.3f},{world[2]:.3f}) "
                    f"sim_obj={np.round(sim_obj,3) if sim_obj is not None else None}"
                )

                cv2.circle(
                    overlay,
                    (int(u), int(v)),
                    6,
                    (255, 0, 0),
                    -1,
                )

        panel = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLO depth world debug", panel)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if done:
            obs = env.reset()

        time.sleep(0.01)

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
