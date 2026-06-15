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
from src.living_room.place_controller import move_to
from src.living_room.ur5e_gripper_controller import ur5e_gripper_command


def get_rgb(obs, camera):
    img = np.asarray(obs[f"{camera}_image"])
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.flipud(img)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def get_depth(obs, env, camera):
    depth = np.asarray(obs[f"{camera}_depth"])
    depth = np.squeeze(depth)
    depth = np.flipud(depth)
    return get_real_depth_map(env.sim, depth)


def pixel_to_world(env, camera, u, v, depth, height, width):
    K = get_camera_intrinsic_matrix(env.sim, camera, height, width)
    E = get_camera_extrinsic_matrix(env.sim, camera)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    cam_point = np.array([x, y, z, 1.0], dtype=np.float64)
    world_point = E @ cam_point
    return world_point[:3]


def ur5e_grip_relative(stage, age):
    # UR5e Robotiq in this robosuite setup is binary-ish:
    # negative opens, positive fully closes, 0 is passive.
    open_cmd = float(os.environ.get("UR5E_OPEN", "-1.0"))
    close_cmd = float(os.environ.get("UR5E_CLOSE", "1.0"))
    hold_cmd = float(os.environ.get("UR5E_HOLD", "0.0"))

    if stage in ["reach_above", "pregrasp", "descend"]:
        return open_cmd

    if stage == "close_gripper":
        # Very short full-close pulse only.
        pulse_steps = int(os.environ.get("UR5E_CLOSE_PULSE_STEPS", "5"))
        if age < pulse_steps:
            return close_cmd
        return hold_cmd

    if stage in ["verify_lift", "hold_lift"]:
        return hold_cmd

    return hold_cmd



def ur5e_object_q_profile(object_type):
    """
    Object-aware Robotiq aperture ranges.

    q is not true metric width, but from probes:
      open q ~ -0.05
      partial close q ~ 0.15
      stronger close q ~ 0.25-0.35
      overclose/instability happens if q drifts too far

    These are starting profiles. We will tune them per object.
    """
    object_type = str(object_type).lower()

    profiles = {
        # Milk box needs stronger closure than 0.15.
        "milk":   (0.240, 0.300),

        # Can is narrower/round, should not overcrush.
        "can":    (0.180, 0.250),

        # Bread is wider/flat-ish, keep gentler.
        "bread":  (0.120, 0.190),

        # Cereal box is bigger, allow more closure.
        "cereal": (0.260, 0.340),
    }

    q_low, q_high = profiles.get(object_type, (0.180, 0.260))

    # Allow env override while debugging.
    q_low = float(os.environ.get("UR5E_Q_LOW", str(q_low)))
    q_high = float(os.environ.get("UR5E_Q_HIGH", str(q_high)))

    return q_low, q_high


def ur5e_gripper_aperture_cmd(obs, stage, object_type):
    """
    Object-aware UR5e Robotiq aperture feedback.

    negative = open
    positive = close
    0.0 = passive

    Instead of fixed q for all objects, choose q range by object type.
    """
    if stage in ["reach_above", "pregrasp", "descend"]:
        return -1.0

    gq = obs.get("robot0_gripper_qpos", None)
    if gq is None:
        return 0.0

    q = float((gq[0] + gq[3]) / 2.0) if len(gq) >= 4 else float(gq[0])
    q_low, q_high = ur5e_object_q_profile(object_type)

    if stage in ["close_gripper", "verify_lift", "hold_lift"]:
        if q < q_low:
            return 1.0       # close more for this object
        if q > q_high:
            return -1.0      # prevent over-close / slipping / object launch
        return 0.0

    return 0.0


def detect_object_world(obs, env, detector, camera, height, width):
    rgb = get_rgb(obs, camera)
    depth_map = get_depth(obs, env, camera)

    detections = detector.detect(rgb)
    if not detections:
        return None, rgb, []

    det = detections[0]
    u, v = det.center_px

    mask_depths = depth_map[det.mask > 0]
    mask_depths = mask_depths[np.isfinite(mask_depths)]
    mask_depths = mask_depths[mask_depths > 0]

    if len(mask_depths) == 0:
        return None, rgb, detections

    d = float(np.median(mask_depths))
    world = pixel_to_world(env, camera, u, v, d, height, width)

    return world, rgb, detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="UR5e")
    parser.add_argument("--object_type", default="milk")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--model", default="runs/segment/train/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=520)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ["ROBOT_NAME"] = args.robot

    # Sweep-safe UR5e defaults.
    os.environ.setdefault("UR5E_OPEN", "-0.20")
    os.environ.setdefault("UR5E_CLOSE_PEAK", "0.030")
    os.environ.setdefault("UR5E_HOLD", "0.004")

    # YOLO depth sees visible surface slightly above sim object center.
    z_correction = float(os.environ.get("UR5E_YOLO_Z_CORRECTION", "0.015"))
    grasp_z_add = float(os.environ.get("UR5E_GRASP_Z_ADD", "0.020"))

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

    print("\n=== UR5e YOLO-depth pickup-only ===")
    print("Detecting object target from YOLO mask...\n")

    world_samples = []

    for i in range(20):
        obs, _, _, _ = env.step(np.zeros(env.action_dim, dtype=np.float32))
        world, rgb, detections = detect_object_world(
            obs, env, detector, args.camera, args.height, args.width
        )
        if world is not None:
            world_samples.append(world)

    if not world_samples:
        print("No YOLO world target found. Stop.")
        env.close()
        return

    yolo_world = np.median(np.stack(world_samples, axis=0), axis=0)
    obj_z0 = float(yolo_world[2] - z_correction)

    target_xy = yolo_world[:2].copy()

    # UR5e Robotiq grasp calibration.
    # YOLO gives object center, but UR5e wrist/eef frame is not the true finger center.
    # Bias lets us align the actual fingers around the object.
    target_xy[0] += float(os.environ.get("UR5E_YOLO_GRASP_BIAS_X", "0.0"))
    target_xy[1] += float(os.environ.get("UR5E_YOLO_GRASP_BIAS_Y", "0.0"))

    print("YOLO world:", np.round(yolo_world, 4))
    print("Corrected object z0:", round(obj_z0, 4))
    print("Target XY:", np.round(target_xy, 4))

    if "Milk_pos" in obs:
        print("Sim object:", np.round(obs["Milk_pos"], 4))

    max_lift = 0.0
    obj_start_z = float(obs["Milk_pos"][2]) if "Milk_pos" in obs else obj_z0

    phase = "reach_above"
    phase_start = 0
    close_start = None

    for step in range(args.horizon):
        env.render()

        eef = obs["robot0_eef_pos"]

        above = np.array([target_xy[0], target_xy[1], obj_z0 + 0.245], dtype=np.float32)
        pregrasp = np.array([target_xy[0], target_xy[1], obj_z0 + 0.145], dtype=np.float32)
        grasp = np.array([target_xy[0], target_xy[1], obj_z0 + grasp_z_add], dtype=np.float32)
        lift = np.array([target_xy[0], target_xy[1], obj_z0 + 0.220], dtype=np.float32)

        # Distance-gated state machine.
        # Do not close until EEF is actually near grasp height.
        xy_err = float(np.linalg.norm(eef[:2] - target_xy))
        z_err_grasp = float(abs(eef[2] - grasp[2]))

        if phase == "reach_above":
            if float(np.linalg.norm(eef - above)) < 0.045 or (step - phase_start) > 120:
                phase = "pregrasp"
                phase_start = step

        elif phase == "pregrasp":
            if float(np.linalg.norm(eef - pregrasp)) < 0.040 or (step - phase_start) > 140:
                phase = "descend"
                phase_start = step

        elif phase == "descend":
            # UR5e may stall around the contact zone because the Robotiq fingers / table / object
            # block further downward motion. Do not wait forever for exact grasp_z.
            contact_zone = eef[2] <= (obj_z0 + float(os.environ.get("UR5E_CLOSE_Z_ABOVE_OBJ", "0.040")))
            descent_timeout = (step - phase_start) > int(os.environ.get("UR5E_DESCEND_TIMEOUT", "260"))

            if xy_err < 0.035 and (contact_zone or descent_timeout):
                phase = "close_gripper"
                phase_start = step
                close_start = step

        elif phase == "close_gripper":
            if close_start is not None and (step - close_start) > 12:
                phase = "verify_lift"
                phase_start = step

        elif phase == "verify_lift":
            if (step - phase_start) > 180:
                phase = "hold_lift"
                phase_start = step

        stage = phase

        if stage == "reach_above":
            target, gain, max_step = above, 4.0, 0.25
            grip = ur5e_grip_relative(stage, 0)
        elif stage == "pregrasp":
            target, gain, max_step = pregrasp, 3.0, 0.15
            grip = ur5e_grip_relative(stage, 0)
        elif stage == "descend":
            target, gain, max_step = grasp, 3.2, 0.050
            grip = ur5e_grip_relative(stage, 0)
        elif stage == "close_gripper":
            target, gain, max_step = eef.copy(), 0.0, 0.0
            age = 0 if close_start is None else step - close_start
            grip = ur5e_grip_relative(stage, age)
        elif stage == "verify_lift":
            target, gain, max_step = lift, 1.4, 0.045
            grip = ur5e_grip_relative(stage, 0)
        else:
            target, gain, max_step = lift, 0.8, 0.03
            grip = ur5e_grip_relative("hold_lift", 0)

        # UR5e final gripper command uses aperture feedback.
        grip = ur5e_gripper_aperture_cmd(obs, stage, args.object_type)

        action = move_to(
            obs=obs,
            target=target,
            gripper=grip,
            action_dim=env.action_dim,
            gain=gain,
            max_step=max_step,
        )

        obs, _, done, _ = env.step(action)

        obj = obs["Milk_pos"] if "Milk_pos" in obs else None
        if obj is not None:
            lift_delta = float(obj[2] - obj_start_z)
            max_lift = max(max_lift, lift_delta)

            if step % 20 == 0 or stage == "close_gripper":
                gq = obs.get("robot0_gripper_qpos", None)
                gq_txt = np.round(gq[:2], 3) if gq is not None else None
                print(
                    f"step={step:03d} stage={stage:13s} grip={grip:.2f} "
                    f"gq={gq_txt} "
                    f"obj={np.round(obj,3)} eef={np.round(obs['robot0_eef_pos'],3)} "
                    f"lift={lift_delta:.3f} max_lift={max_lift:.3f}"
                )

        # Draw live YOLO overlay.
        world_now, rgb, detections = detect_object_world(
            obs, env, detector, args.camera, args.height, args.width
        )
        overlay = draw_yolo_detections(rgb, detections)
        panel = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.putText(
            panel,
            f"stage={stage} max_lift={max_lift:.3f}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("UR5e YOLO pickup-only", panel)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if done:
            break

        if obj is not None:
            if np.linalg.norm(obj[:2]) > 1.25 or obj[2] < 0.05:
                print("ABORT: object unstable")
                break

        time.sleep(0.01)

    print("\nDone.")
    print("pickup_success:", bool(max_lift > 0.08))
    print("max_lift:", max_lift)

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
