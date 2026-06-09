import os
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import robosuite as suite

from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
    get_real_depth_map,
)

from src.living_room.ur5e_yolo_detector import YoloMaskDetector
from src.living_room.place_controller import move_to


os.environ.setdefault("MUJOCO_GL", "glfw")


PHASE_ID = {
    "reach_above": 0,
    "pregrasp": 1,
    "descend": 2,
    "side_pregrasp": 6,
    "side_approach": 7,
    "close_gripper": 3,
    "verify_lift": 4,
    "hold_lift": 5,
}


def get_object_pos(env, obs, object_type):
    """
    Robust object position getter for robosuite PickPlace.

    The previous version accidentally returned [0, 0, 0] from a non-object body,
    which made every rollout abort immediately.
    """
    obj = object_type.lower()

    # 1. Prefer object observations, e.g. Milk0_pos / milk_pos.
    for k, v in obs.items():
        kl = k.lower()
        arr = np.asarray(v)

        if arr.shape == (3,) and kl.endswith("pos") and obj in kl:
            pos = arr.astype(np.float32).copy()
            if 0.60 < float(pos[2]) < 1.30:
                return pos

    # 2. Try robosuite obj_body_id dictionary.
    body_ids = getattr(env, "obj_body_id", None)
    if isinstance(body_ids, dict):
        for name, bid in body_ids.items():
            if obj in str(name).lower():
                pos = env.sim.data.body_xpos[int(bid)].astype(np.float32).copy()
                if 0.60 < float(pos[2]) < 1.30:
                    return pos

    # 3. Try env.objects names.
    for robj in getattr(env, "objects", []):
        name = getattr(robj, "name", "")
        if obj in str(name).lower():
            for i in range(env.sim.model.nbody):
                bname = env.sim.model.body_id2name(i)
                if bname and str(name).lower() in bname.lower():
                    pos = env.sim.data.body_xpos[i].astype(np.float32).copy()
                    if 0.60 < float(pos[2]) < 1.30:
                        return pos

    # 4. Last fallback: body names containing object type, but reject [0,0,0].
    best = None
    best_score = 1e9

    for i in range(env.sim.model.nbody):
        name = env.sim.model.body_id2name(i)
        if not name:
            continue

        nl = name.lower()
        if obj not in nl:
            continue

        pos = env.sim.data.body_xpos[i].astype(np.float32).copy()

        # reject world/root/default-zero bodies
        if not (0.60 < float(pos[2]) < 1.30):
            continue

        # object should be near tabletop height around 0.88
        score = abs(float(pos[2]) - 0.88)
        if score < best_score:
            best = pos
            best_score = score

    if best is not None:
        return best

    print("Available obs pos keys:")
    for k, v in obs.items():
        arr = np.asarray(v)
        if "pos" in k.lower():
            print(" ", k, arr.shape, np.round(arr, 4))

    print("Candidate body names:")
    for i in range(env.sim.model.nbody):
        name = env.sim.model.body_id2name(i)
        if name and obj in name.lower():
            print(" ", i, name, np.round(env.sim.data.body_xpos[i], 4))

    raise RuntimeError(f"Could not find valid object position for {object_type}")


def sim_state(env):
    return np.concatenate([
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
    ]).astype(np.float32)


def _geom_label(env, gid):
    gid = int(gid)

    try:
        gname = env.sim.model.geom_id2name(gid)
    except Exception:
        gname = None

    try:
        bid = int(env.sim.model.geom_bodyid[gid])
    except Exception:
        bid = -1

    try:
        bname = env.sim.model.body_id2name(bid)
    except Exception:
        bname = None

    if gname is None:
        gname = f"geom{gid}"
    if bname is None:
        bname = f"body{bid}"

    return f"{gname}[body={bname}, gid={gid}]"


def contact_pairs(env, object_type, max_pairs=30):
    pairs = []
    obj = object_type.lower()
    keywords = [obj, "milk", "can", "bread", "cereal", "gripper", "finger", "robotiq", "wrist", "eef", "right", "left"]

    try:
        ncon = int(env.sim.data.ncon)
    except Exception:
        return pairs

    for i in range(ncon):
        c = env.sim.data.contact[i]
        g1 = _geom_label(env, c.geom1)
        g2 = _geom_label(env, c.geom2)

        pair = f"{g1}<->{g2}"
        low = pair.lower()

        # Always show object contacts, plus gripper/finger contacts.
        if any(k in low for k in keywords):
            pairs.append(pair)

    return pairs[:max_pairs]


def gripper_body_positions(env, obj_pos, max_items=12):
    rows = []

    try:
        nbody = int(env.sim.model.nbody)
    except Exception:
        return rows

    for bid in range(nbody):
        try:
            name = env.sim.model.body_id2name(bid)
        except Exception:
            continue

        if name is None:
            continue

        low = str(name).lower()
        if "gripper0" in low and ("finger" in low or "knuckle" in low):
            pos = env.sim.data.body_xpos[bid].copy()
            d = pos - obj_pos
            xy = float(np.linalg.norm(d[:2]))
            rows.append((xy, str(name), pos, d))

    rows.sort(key=lambda x: x[0])

    out = []
    for xy, name, pos, d in rows[:max_items]:
        out.append(
            f"{name}: pos={np.round(pos,3)} d_obj={np.round(d,3)} xy={xy:.3f}"
        )
    return out


def pixel_to_world(env, camera, height, width, px, py, depth_m):
    K = get_camera_intrinsic_matrix(env.sim, camera, height, width)
    E = get_camera_extrinsic_matrix(env.sim, camera)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (px - cx) * depth_m / fx
    y = (py - cy) * depth_m / fy

    p_cam = np.array([x, y, depth_m, 1.0], dtype=np.float64)
    p_world = E @ p_cam
    return p_world[:3].astype(np.float32)


def detect_object_world(obs, env, detector, camera, height, width):
    image_key = f"{camera}_image"
    depth_key = f"{camera}_depth"

    img = obs[image_key]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    dets = detector.detect(img)
    if len(dets) == 0:
        raise RuntimeError("YOLO found no object mask")

    det = max(dets, key=lambda d: float(d.conf))
    mask = det.mask.astype(bool)

    depth_raw = obs[depth_key]
    depth_real = get_real_depth_map(env.sim, depth_raw)

    vals = depth_real[mask]
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]

    if vals.size == 0:
        raise RuntimeError("No valid depth under YOLO mask")

    depth_m = float(np.median(vals))
    px, py = det.center_px

    world = pixel_to_world(env, camera, height, width, float(px), float(py), depth_m)

    return world, img, det


def object_q_profile(object_type):
    obj = object_type.lower()

    profiles = {
        "milk": (0.24, 0.30),
        "can": (0.18, 0.25),
        "bread": (0.12, 0.19),
        "cereal": (0.26, 0.34),
    }

    return profiles.get(obj, (0.18, 0.26))


def gripper_cmd(obs, phase, object_type, q_low=None, q_high=None):
    if phase in ["reach_above", "pregrasp", "descend", "side_pregrasp", "side_approach"]:
        return -1.0

    gq = obs.get("robot0_gripper_qpos", None)
    if gq is None:
        return 0.0

    q = float((gq[0] + gq[3]) / 2.0) if len(gq) >= 4 else float(gq[0])

    if q_low is None or q_high is None:
        q_low, q_high = object_q_profile(object_type)

    if phase in ["close_gripper", "verify_lift", "hold_lift"]:
        if q < q_low:
            return 1.0
        if q > q_high:
            return -1.0
        return 0.0

    return 0.0


def run_trial(args, detector, trial_id):
    env = suite.make(
        env_name="PickPlace",
        robots=args.robot,
        gripper_types=args.gripper,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=[args.camera],
        camera_heights=args.height,
        camera_widths=args.width,
        camera_depths=True,
        single_object_mode=2,
        object_type=args.object_type,
        ignore_done=True,
        horizon=args.horizon,
    )

    obs = env.reset()

    yolo_world, first_img, det = detect_object_world(
        obs, env, detector, args.camera, args.height, args.width
    )

    obj0 = get_object_pos(env, obs, args.object_type)

    # Teacher target source:
    # For collecting demos, use privileged sim object pose.
    # We still save YOLO world and images for learning/debugging.
    if args.teacher_source == "sim":
        obj_z0 = float(obj0[2])
        target_xy = obj0[:2].copy()
    else:
        obj_z0 = float(yolo_world[2] - args.yolo_z_correction)
        target_xy = yolo_world[:2].copy()

    target_xy[0] += args.bias_x
    target_xy[1] += args.bias_y

    phase = "side_pregrasp" if args.grasp_mode == "side" else "reach_above"
    phase_start = 0
    close_start = None

    max_lift = 0.0
    unstable = False

    actions = []
    states = []
    images = []
    eef_pos = []
    grip_qpos = []
    object_pos = []
    yolo_worlds = []
    target_xys = []
    phase_ids = []
    rewards = []
    dones = []

    print("\n=== trial", trial_id, "===")
    print("YOLO world:", np.round(yolo_world, 4))
    print("Target XY:", np.round(target_xy, 4))
    print("Sim object:", np.round(obj0, 4))

    for step in range(args.horizon):
        obj = get_object_pos(env, obs, args.object_type)
        eef = obs["robot0_eef_pos"]

        lift = float(obj[2] - obj0[2])
        max_lift = max(max_lift, lift)

        above = np.array([target_xy[0], target_xy[1], obj_z0 + 0.24], dtype=np.float32)
        pregrasp = np.array([target_xy[0], target_xy[1], obj_z0 + 0.13], dtype=np.float32)
        grasp = np.array([target_xy[0], target_xy[1], obj_z0 + args.grasp_z_add], dtype=np.float32)

        # Side-grasp geometry.
        side_dir = np.array([1.0, 0.0], dtype=np.float32)
        if args.side == "x_minus":
            side_dir = np.array([-1.0, 0.0], dtype=np.float32)
        elif args.side == "x_plus":
            side_dir = np.array([1.0, 0.0], dtype=np.float32)
        elif args.side == "y_minus":
            side_dir = np.array([0.0, -1.0], dtype=np.float32)
        elif args.side == "y_plus":
            side_dir = np.array([0.0, 1.0], dtype=np.float32)

        side_z = obj_z0 + args.side_z_add
        side_pre_xy = target_xy + side_dir * args.side_pre_dist
        side_contact_xy = target_xy + side_dir * args.side_contact_dist

        side_pre = np.array([side_pre_xy[0], side_pre_xy[1], side_z], dtype=np.float32)
        side_contact = np.array([side_contact_xy[0], side_contact_xy[1], side_z], dtype=np.float32)
        lift_target = np.array([target_xy[0], target_xy[1], obj_z0 + 0.23], dtype=np.float32)

        if phase == "side_pregrasp":
            target = side_pre
            gain = 5.0
            max_step = 0.080

            # Do not switch early. Must actually reach side pregrasp.
            if np.linalg.norm(eef - side_pre) < 0.040:
                phase = "side_approach"
                phase_start = step

        elif phase == "side_approach":
            target = side_contact
            gain = 4.0
            max_step = 0.050

            # Do not close unless gripper is actually at the side-contact pose.
            if np.linalg.norm(eef - side_contact) < args.side_close_thresh:
                phase = "close_gripper"
                phase_start = step
                close_start = step

        elif phase == "reach_above":
            target = above
            gain = 4.0
            max_step = 0.055
            if np.linalg.norm(eef[:2] - target_xy) < 0.035:
                phase = "pregrasp"
                phase_start = step

        elif phase == "pregrasp":
            target = pregrasp
            gain = 3.0
            max_step = 0.035
            if abs(float(eef[2] - pregrasp[2])) < 0.035 or (step - phase_start) > 90:
                phase = "descend"
                phase_start = step

        elif phase == "descend":
            target = grasp
            gain = 5.0
            max_step = 0.060

            contact_zone = eef[2] <= (obj_z0 + args.close_z_above_obj)
            timeout = (step - phase_start) > args.descend_timeout
            near_object_z = eef[2] <= (obj_z0 + 0.055)

            if np.linalg.norm(eef[:2] - target_xy) < 0.040 and (contact_zone or (timeout and near_object_z)):
                phase = "close_gripper"
                phase_start = step
                close_start = step

        elif phase == "close_gripper":
            target = side_contact if args.grasp_mode == "side" else grasp
            gain = 1.5
            max_step = 0.010

            age = step - close_start
            if age > args.close_max_steps:
                phase = "verify_lift"
                phase_start = step

        elif phase == "verify_lift":
            target = lift_target
            gain = 2.5
            max_step = 0.025

            if (step - phase_start) > 180:
                phase = "hold_lift"
                phase_start = step

        else:
            target = lift_target
            gain = 2.0
            max_step = 0.015

        grip = gripper_cmd(obs, phase, args.object_type, args.q_low, args.q_high)

        action = move_to(
            obs=obs,
            target=target,
            gripper=grip,
            action_dim=env.action_dim,
            gain=gain,
            max_step=max_step,
        )

        img = obs[f"{args.camera}_image"]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        actions.append(action.astype(np.float32))
        states.append(sim_state(env))
        images.append(img.astype(np.uint8))
        eef_pos.append(np.asarray(eef, dtype=np.float32))
        grip_qpos.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
        object_pos.append(np.asarray(obj, dtype=np.float32))
        yolo_worlds.append(yolo_world.astype(np.float32))
        target_xys.append(target_xy.astype(np.float32))
        phase_ids.append(PHASE_ID[phase])

        obs, reward, done, info = env.step(action)

        rewards.append(float(reward))
        dones.append(False)

        if args.render:
            env.render()
            time.sleep(args.sleep)

        if step % 20 == 0 or phase == "close_gripper":
            gq = obs.get("robot0_gripper_qpos", None)
            gq_txt = np.round(gq[:2], 3) if gq is not None else None
            print(
                f"step={step:03d} phase={phase:13s} grip={grip:+.1f} "
                f"gq={gq_txt} obj={np.round(obj,3)} eef={np.round(eef,3)} "
                f"lift={lift:.3f} max={max_lift:.3f}"
            )

            if args.print_contacts and phase in ["close_gripper", "verify_lift", "hold_lift"]:
                pairs = contact_pairs(env, args.object_type)
                if pairs:
                    print("  contacts:", pairs)
                else:
                    print("  contacts: NONE")

                if args.print_fingers:
                    for line in gripper_body_positions(env, obj):
                        print("  finger:", line)

        xy_move = float(np.linalg.norm(obj[:2] - obj0[:2]))
        if xy_move > args.unstable_xy or lift > args.unstable_z:
            unstable = True
            print("ABORT unstable:", "xy_move", round(xy_move, 3), "lift", round(lift, 3))
            break

    final_obj = get_object_pos(env, obs, args.object_type)
    final_eef = obs["robot0_eef_pos"]
    final_lift = float(final_obj[2] - obj0[2])
    final_xy_dist = float(np.linalg.norm(final_obj[:2] - final_eef[:2]))

    success = (
        (max_lift > args.success_lift)
        and (not unstable)
        and (final_xy_dist < args.success_xy_dist)
    )

    if len(dones) > 0:
        dones[-1] = True

    result = {
        "actions": np.asarray(actions, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float32),
        "agentview_image": np.asarray(images, dtype=np.uint8),
        "robot0_eef_pos": np.asarray(eef_pos, dtype=np.float32),
        "robot0_gripper_qpos": np.asarray(grip_qpos, dtype=np.float32),
        "object_pos": np.asarray(object_pos, dtype=np.float32),
        "yolo_world": np.asarray(yolo_worlds, dtype=np.float32),
        "target_xy": np.asarray(target_xys, dtype=np.float32),
        "phase_id": np.asarray(phase_ids, dtype=np.int32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "success": bool(success),
        "unstable": bool(unstable),
        "max_lift": float(max_lift),
        "final_lift": float(final_lift),
        "final_xy_dist": float(final_xy_dist),
    }

    print(
        "result:",
        "success", success,
        "unstable", unstable,
        "max_lift", round(max_lift, 4),
        "final_xy_dist", round(final_xy_dist, 4),
    )

    env.close()
    return result


def write_dataset(path, trials, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env"] = "PickPlace"
        data.attrs["robot"] = args.robot
        data.attrs["object_type"] = args.object_type
        data.attrs["camera"] = args.camera

        total = 0
        for i, tr in enumerate(trials):
            grp = data.create_group(f"demo_{i}")

            grp.create_dataset("actions", data=tr["actions"])
            grp.create_dataset("states", data=tr["states"])
            grp.create_dataset("rewards", data=tr["rewards"])
            grp.create_dataset("dones", data=tr["dones"])

            obs_grp = grp.create_group("obs")
            obs_grp.create_dataset("agentview_image", data=tr["agentview_image"], compression="gzip")
            obs_grp.create_dataset("robot0_eef_pos", data=tr["robot0_eef_pos"])
            obs_grp.create_dataset("robot0_gripper_qpos", data=tr["robot0_gripper_qpos"])
            obs_grp.create_dataset("object_pos", data=tr["object_pos"])
            obs_grp.create_dataset("yolo_world", data=tr["yolo_world"])
            obs_grp.create_dataset("target_xy", data=tr["target_xy"])
            obs_grp.create_dataset("phase_id", data=tr["phase_id"])

            grp.attrs["num_samples"] = tr["actions"].shape[0]
            grp.attrs["success"] = int(tr["success"])
            grp.attrs["unstable"] = int(tr["unstable"])
            grp.attrs["max_lift"] = tr["max_lift"]
            grp.attrs["final_lift"] = tr["final_lift"]
            grp.attrs["final_xy_dist"] = tr["final_xy_dist"]

            total += tr["actions"].shape[0]

        data.attrs["total"] = total

    print("Wrote:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="UR5e")
    parser.add_argument("--gripper", default=None)
    parser.add_argument("--object_type", default="milk")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="datasets/ur5e_robomimic/raw/ur5e_milk_pick_rollouts.hdf5")

    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--conf", type=float, default=0.25)

    parser.add_argument("--grasp_z_add", type=float, default=-0.010)
    parser.add_argument("--yolo_z_correction", type=float, default=0.015)
    parser.add_argument("--close_z_above_obj", type=float, default=0.015)
    parser.add_argument("--descend_timeout", type=int, default=420)
    parser.add_argument("--close_max_steps", type=int, default=45)

    parser.add_argument("--bias_x", type=float, default=0.0)
    parser.add_argument("--bias_y", type=float, default=0.0)
    parser.add_argument("--q_low", type=float, default=None)
    parser.add_argument("--q_high", type=float, default=None)
    parser.add_argument("--teacher_source", choices=["sim", "yolo"], default="sim")
    parser.add_argument("--grasp_mode", choices=["top", "side"], default="side")
    parser.add_argument("--side", choices=["x_minus", "x_plus", "y_minus", "y_plus"], default="x_minus")
    parser.add_argument("--side_pre_dist", type=float, default=0.105)
    parser.add_argument("--side_contact_dist", type=float, default=0.035)
    parser.add_argument("--side_z_add", type=float, default=0.030)
    parser.add_argument("--side_close_thresh", type=float, default=0.008)

    parser.add_argument("--success_lift", type=float, default=0.06)
    parser.add_argument("--success_xy_dist", type=float, default=0.12)
    parser.add_argument("--unstable_xy", type=float, default=0.25)
    parser.add_argument("--unstable_z", type=float, default=0.30)

    parser.add_argument("--print_contacts", action="store_true")
    parser.add_argument("--print_fingers", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)

    args = parser.parse_args()

    detector = YoloMaskDetector(args.model, conf=args.conf)

    trials = []
    for i in range(args.num_trials):
        try:
            tr = run_trial(args, detector, i)
            trials.append(tr)
        except Exception as e:
            print("TRIAL FAILED:", i, repr(e))

    write_dataset(args.out, trials, args)

    n_success = sum(int(t["success"]) for t in trials)
    print(f"Summary: {n_success}/{len(trials)} stable successes")


if __name__ == "__main__":
    main()
