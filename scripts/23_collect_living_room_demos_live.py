import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--demos_per_task", type=int, default=3)
parser.add_argument("--max_attempts_per_task", type=int, default=6)
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--horizon", type=int, default=270)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--out", type=str, default="datasets/living_room/living_room_demos_v0.hdf5")
parser.add_argument("--sleep", type=float, default=0.005)
args = parser.parse_args()

os.environ["MUJOCO_GL"] = args.gl

import h5py
import imageio
import numpy as np

from src.envs.render_utils import show_camera_panel, close_render_windows
from src.living_room.pickplace_env import make_pickplace_env, get_bin_pos


OBJECTS = ["Milk", "Bread", "Cereal", "Can"]
STAGE_TO_ID = {
    "reach_above": 0,
    "pregrasp": 1,
    "descend": 2,
    "close_gripper": 3,
    "lift": 4,
    "move_to_tray": 5,
    "lower_to_tray": 6,
    "release": 7,
    "retreat": 8,
}


def load_tasks():
    cfg = json.loads(Path("configs/living_room/tasks_v0.json").read_text())
    return cfg["tasks"], cfg


def move_to(obs, target, gripper, action_dim, gain=8.0, max_step=0.8):
    eef = obs["robot0_eef_pos"]
    delta = target - eef

    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = np.clip(delta * gain, -max_step, max_step)

    if action_dim >= 6:
        action[3:6] = 0.0

    action[-1] = gripper
    return action


def object_grasp_offset(obj_name):
    if obj_name == "Milk":
        return 0.015
    if obj_name == "Bread":
        return 0.010
    if obj_name == "Can":
        return 0.010
    if obj_name == "Cereal":
        return 0.020
    return 0.015


def scripted_policy(obs, step, env, obj_name, obj_start, bin_pos):
    open_gripper = -1.0
    close_gripper = 1.0

    obj = obs[f"{obj_name}_pos"].copy()
    grasp_offset = object_grasp_offset(obj_name)

    above_obj = np.array([obj[0], obj[1], obj_start[2] + 0.18])
    pregrasp = np.array([obj[0], obj[1], obj_start[2] + 0.07])
    grasp = np.array([obj[0], obj[1], obj_start[2] + grasp_offset])
    carry = np.array([obj[0], obj[1], obj_start[2] + 0.28])

    above_bin = np.array([bin_pos[0], bin_pos[1], obj_start[2] + 0.30])
    drop = np.array([bin_pos[0], bin_pos[1], obj_start[2] + 0.12])
    retreat = np.array([bin_pos[0], bin_pos[1], obj_start[2] + 0.30])

    if step < 40:
        stage, target, grip, gain = "reach_above", above_obj, open_gripper, 9.0
    elif step < 80:
        stage, target, grip, gain = "pregrasp", pregrasp, open_gripper, 8.0
    elif step < 115:
        stage, target, grip, gain = "descend", grasp, open_gripper, 7.0
    elif step < 155:
        stage, target, grip, gain = "close_gripper", grasp, close_gripper, 5.0
    elif step < 185:
        stage, target, grip, gain = "lift", carry, close_gripper, 6.0
    elif step < 220:
        stage, target, grip, gain = "move_to_tray", above_bin, close_gripper, 5.0
    elif step < 240:
        stage, target, grip, gain = "lower_to_tray", drop, close_gripper, 5.0
    elif step < 255:
        stage, target, grip, gain = "release", drop, open_gripper, 4.0
    else:
        stage, target, grip, gain = "retreat", retreat, open_gripper, 5.0

    action = move_to(obs, target, grip, env.action_dim, gain=gain)
    return action, stage


def success_check(obs, obj_name, bin_pos):
    obj = obs[f"{obj_name}_pos"]
    xy_dist = np.linalg.norm(obj[:2] - bin_pos[:2])
    z_ok = obj[2] > 0.78
    return bool(xy_dist < 0.12 and z_ok), xy_dist


def collect_one_attempt(task, cfg, attempt_id):
    obj_name = task["robosuite_object"]
    alias = task["living_room_object"]
    instruction = task["instruction"]
    task_id = task["task_id"]

    env, camera_names = make_pickplace_env(
        render=True,
        camera_height=args.height,
        camera_width=args.width,
        horizon=args.horizon,
    )

    obs = env.reset()

    camera_keys = [f"{name}_image" for name in camera_names]
    obj_start = obs[f"{obj_name}_pos"].copy()
    bin_pos = get_bin_pos(env, task["target_container"])

    ep = {
        "agentview_image": [],
        "frontview_image": [],
        "actions": [],
        "eef_pos": [],
        "eef_quat": [],
        "target_object_pos": [],
        "bin_pos": [],
        "stage_ids": [],
        "stage_names": [],
        "rewards": [],
        "dones": [],
        "all_object_pos": [],
    }

    frames = []
    success = False
    best_xy = 999.0

    for step in range(args.horizon):
        action, stage = scripted_policy(
            obs=obs,
            step=step,
            env=env,
            obj_name=obj_name,
            obj_start=obj_start,
            bin_pos=bin_pos,
        )

        obs, reward, done, info = env.step(action)

        ok, xy_dist = success_check(obs, obj_name, bin_pos)
        best_xy = min(best_xy, xy_dist)

        env.render()

        all_obj_pos = []
        for name in OBJECTS:
            all_obj_pos.append(obs[f"{name}_pos"])

        ep["agentview_image"].append(obs["agentview_image"])
        ep["frontview_image"].append(obs["frontview_image"])
        ep["actions"].append(action)
        ep["eef_pos"].append(obs["robot0_eef_pos"])
        ep["eef_quat"].append(obs["robot0_eef_quat"])
        ep["target_object_pos"].append(obs[f"{obj_name}_pos"])
        ep["bin_pos"].append(bin_pos)
        ep["stage_ids"].append(STAGE_TO_ID[stage])
        ep["stage_names"].append(stage.encode("utf-8"))
        ep["rewards"].append(reward)
        ep["dones"].append(done)
        ep["all_object_pos"].append(np.asarray(all_obj_pos, dtype=np.float32))

        frames.append(obs["agentview_image"][::-1])

        title = (
            f"{task_id} | {instruction} | attempt={attempt_id} "
            f"step={step} stage={stage} xy={xy_dist:.3f}"
        )

        stop = show_camera_panel(
            obs=obs,
            camera_keys=camera_keys,
            window_name="Mini-RT2-Room Demo Collection",
            title_text=title,
            delay=1,
        )

        if step % 25 == 0:
            print(
                f"{task_id} attempt={attempt_id} step={step:03d} "
                f"stage={stage:<14} xy={xy_dist:.3f} "
                f"obj={np.round(obs[f'{obj_name}_pos'], 3)}"
            )

        if ok and step > 230:
            success = True
            print(f"{task_id} SUCCESS step={step} best_xy={best_xy:.4f}")
            break

        if done or stop:
            break

        time.sleep(args.sleep)

    video_path = f"videos/living_room/demo_{task_id}_attempt_{attempt_id:04d}.mp4"
    imageio.mimsave(video_path, frames, fps=20)

    env.close()
    close_render_windows()

    return success, best_xy, video_path, ep


def save_demo(f, demo_name, task, success, best_xy, video_path, ep):
    g = f.create_group(demo_name)

    g.attrs["task_id"] = task["task_id"]
    g.attrs["instruction"] = task["instruction"]
    g.attrs["robosuite_object"] = task["robosuite_object"]
    g.attrs["living_room_object"] = task["living_room_object"]
    g.attrs["target_container"] = task["target_container"]
    g.attrs["success"] = bool(success)
    g.attrs["best_xy_to_bin"] = float(best_xy)
    g.attrs["video_path"] = video_path

    g.create_dataset("agentview_image", data=np.asarray(ep["agentview_image"], dtype=np.uint8), compression="gzip")
    g.create_dataset("frontview_image", data=np.asarray(ep["frontview_image"], dtype=np.uint8), compression="gzip")
    g.create_dataset("actions", data=np.asarray(ep["actions"], dtype=np.float32))
    g.create_dataset("eef_pos", data=np.asarray(ep["eef_pos"], dtype=np.float32))
    g.create_dataset("eef_quat", data=np.asarray(ep["eef_quat"], dtype=np.float32))
    g.create_dataset("target_object_pos", data=np.asarray(ep["target_object_pos"], dtype=np.float32))
    g.create_dataset("bin_pos", data=np.asarray(ep["bin_pos"], dtype=np.float32))
    g.create_dataset("all_object_pos", data=np.asarray(ep["all_object_pos"], dtype=np.float32))
    g.create_dataset("stage_ids", data=np.asarray(ep["stage_ids"], dtype=np.int64))
    g.create_dataset("stage_names", data=np.asarray(ep["stage_names"]))
    g.create_dataset("rewards", data=np.asarray(ep["rewards"], dtype=np.float32))
    g.create_dataset("dones", data=np.asarray(ep["dones"], dtype=np.bool_))


def main():
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("videos/living_room").mkdir(parents=True, exist_ok=True)

    tasks, cfg = load_tasks()

    print("Collecting Mini-RT2-Room demos")
    print("Output:", args.out)
    print("Demos per task:", args.demos_per_task)
    print("Max attempts per task:", args.max_attempts_per_task)
    print("Live rendering: ON")
    print("GL:", args.gl)

    demo_index = 0
    total_success = 0
    total_attempts = 0

    with h5py.File(args.out, "w") as f:
        f.attrs["project"] = "Mini-RT2-Room"
        f.attrs["base_env"] = "PickPlace"
        f.attrs["robot"] = "Panda"
        f.attrs["camera_names"] = "agentview,frontview"
        f.attrs["action_dim"] = 7
        f.attrs["stage_names"] = ",".join(STAGE_TO_ID.keys())
        f.attrs["object_names"] = ",".join(OBJECTS)

        for task in tasks:
            task_success = 0
            task_attempts = 0

            print()
            print("=" * 80)
            print("Task:", task["task_id"])
            print("Instruction:", task["instruction"])

            while task_success < args.demos_per_task and task_attempts < args.max_attempts_per_task:
                success, best_xy, video_path, ep = collect_one_attempt(
                    task=task,
                    cfg=cfg,
                    attempt_id=total_attempts,
                )

                total_attempts += 1
                task_attempts += 1

                if success:
                    demo_name = f"demo_{demo_index:05d}"
                    save_demo(f, demo_name, task, success, best_xy, video_path, ep)
                    print(f"Saved {demo_name} | task={task['task_id']} | steps={len(ep['actions'])}")
                    demo_index += 1
                    task_success += 1
                    total_success += 1
                else:
                    print(f"Failed attempt not saved | task={task['task_id']} | best_xy={best_xy:.4f}")

            print(f"Task done: {task['task_id']} success={task_success}/{task_attempts}")

        f.attrs["successful_demos"] = total_success
        f.attrs["attempts"] = total_attempts

    print()
    print("Done.")
    print("successful_demos:", total_success)
    print("attempts:", total_attempts)
    print("dataset:", args.out)


if __name__ == "__main__":
    main()
