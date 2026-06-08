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
parser.add_argument("--max_attempts_per_task", type=int, default=10)
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--horizon", type=int, default=480)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--out", type=str, default="datasets/living_room/living_room_multibox_strict_small.hdf5")
parser.add_argument("--sleep", type=float, default=0.005)
args = parser.parse_args()

os.environ["MUJOCO_GL"] = args.gl

import h5py
import imageio
import numpy as np

from src.envs.render_utils import show_camera_panel, close_render_windows
from src.living_room.pickplace_env import make_pickplace_env, get_bin_pos
from src.living_room.place_controller import (
    STAGE_TO_ID,
    scripted_place_policy,
    strict_success,
)

OBJECTS = ["Milk", "Bread", "Cereal", "Can"]


def load_tasks():
    cfg = json.loads(Path("configs/living_room/tasks_v0.json").read_text())
    return cfg["tasks"], cfg


def collect_one_attempt(task, attempt_id):
    obj_name = task["robosuite_object"]
    task_id = task["task_id"]
    instruction = task["instruction"]
    target_container = task["target_container"]

    env, camera_names = make_pickplace_env(
        render=True,
        camera_height=args.height,
        camera_width=args.width,
        horizon=args.horizon,
    )

    obs = env.reset()
    camera_keys = [f"{name}_image" for name in camera_names]

    obj_start = obs[f"{obj_name}_pos"].copy()
    bin_pos = get_bin_pos(env, target_container)

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
    max_lift_delta = 0.0
    success = False
    final_metrics = {}

    print()
    print(f"Attempt {attempt_id} | {task_id}")
    print("Instruction:", instruction)
    print("Object:", obj_name)
    print("Target:", target_container)
    print("Object start:", np.round(obj_start, 3))
    print("Bin pos:", np.round(bin_pos, 3))

    for step in range(args.horizon):
        action, stage = scripted_place_policy(
            obs=obs,
            step=step,
            env=env,
            obj_name=obj_name,
            obj_start=obj_start,
            bin_pos=bin_pos,
        )

        obs, reward, done, info = env.step(action)

        obj_pos = obs[f"{obj_name}_pos"].copy()
        max_lift_delta = max(max_lift_delta, float(obj_pos[2] - obj_start[2]))

        ok, metrics = strict_success(
            obs=obs,
            obj_name=obj_name,
            obj_start=obj_start,
            bin_pos=bin_pos,
            max_lift_delta=max_lift_delta,
            step=step,
        )

        final_metrics = metrics

        all_obj_pos = []
        for name in OBJECTS:
            all_obj_pos.append(obs[f"{name}_pos"])

        ep["agentview_image"].append(obs["agentview_image"])
        ep["frontview_image"].append(obs["frontview_image"])
        ep["actions"].append(action)
        ep["eef_pos"].append(obs["robot0_eef_pos"])
        ep["eef_quat"].append(obs["robot0_eef_quat"])
        ep["target_object_pos"].append(obj_pos)
        ep["bin_pos"].append(bin_pos)
        ep["stage_ids"].append(STAGE_TO_ID[stage])
        ep["stage_names"].append(stage.encode("utf-8"))
        ep["rewards"].append(reward)
        ep["dones"].append(done)
        ep["all_object_pos"].append(np.asarray(all_obj_pos, dtype=np.float32))

        env.render()

        title = (
            f"{task_id} attempt={attempt_id} step={step} stage={stage} "
            f"xy={metrics['final_xy']:.3f} lift={metrics['lift_delta']:.3f}"
        )

        stop = show_camera_panel(
            obs=obs,
            camera_keys=camera_keys,
            window_name="Mini-RT2-Room STRICT Demo Collection",
            title_text=title,
            delay=1,
        )

        frames.append(obs["agentview_image"][::-1])

        if step % 25 == 0:
            print(
                f"{task_id} attempt={attempt_id} step={step:03d} "
                f"stage={stage:<14} "
                f"xy={metrics['final_xy']:.3f} "
                f"z={metrics['final_z']:.3f} "
                f"lift={metrics['lift_delta']:.3f}"
            )

        if ok:
            success = True
            print(
                f"STRICT SUCCESS | step={step} "
                f"xy={metrics['final_xy']:.4f} "
                f"lift={metrics['lift_delta']:.4f}"
            )
            break

        if done or stop:
            break

        time.sleep(args.sleep)

    video_path = f"videos/living_room/strict_demo_{task_id}_attempt_{attempt_id:04d}.mp4"
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=20)

    env.close()
    close_render_windows()

    return success, final_metrics, video_path, ep


def save_demo(f, demo_name, task, success, metrics, video_path, ep):
    g = f.create_group(demo_name)

    g.attrs["task_id"] = task["task_id"]
    g.attrs["instruction"] = task["instruction"]
    g.attrs["robosuite_object"] = task["robosuite_object"]
    g.attrs["living_room_object"] = task["living_room_object"]
    g.attrs["target_container"] = task["target_container"]
    g.attrs["success"] = bool(success)
    g.attrs["final_xy"] = float(metrics["final_xy"])
    g.attrs["final_z"] = float(metrics["final_z"])
    g.attrs["lift_delta"] = float(metrics["lift_delta"])
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

    print("STRICT Mini-RT2-Room demo collection")
    print("Output:", args.out)
    print("Tasks:", len(tasks))
    print("Demos per task:", args.demos_per_task)
    print("Live rendering: ON")
    print("GL:", args.gl)

    demo_index = 0
    total_attempts = 0
    total_success = 0

    with h5py.File(args.out, "w") as f:
        f.attrs["project"] = "Mini-RT2-Room"
        f.attrs["base_env"] = "PickPlace"
        f.attrs["robot"] = "Panda"
        f.attrs["camera_names"] = "agentview,frontview"
        f.attrs["action_dim"] = 7
        f.attrs["stage_names"] = ",".join(STAGE_TO_ID.keys())
        f.attrs["object_names"] = ",".join(OBJECTS)
        f.attrs["strict_success"] = True

        for task in tasks:
            task_success = 0
            task_attempts = 0

            print()
            print("=" * 80)
            print("Task:", task["task_id"])
            print("Instruction:", task["instruction"])

            while task_success < args.demos_per_task and task_attempts < args.max_attempts_per_task:
                success, metrics, video_path, ep = collect_one_attempt(
                    task=task,
                    attempt_id=total_attempts,
                )

                total_attempts += 1
                task_attempts += 1

                if success:
                    demo_name = f"demo_{demo_index:05d}"
                    save_demo(f, demo_name, task, success, metrics, video_path, ep)
                    print(f"Saved {demo_name} | task={task['task_id']} | steps={len(ep['actions'])}")
                    demo_index += 1
                    task_success += 1
                    total_success += 1
                else:
                    print(
                        "Rejected attempt | "
                        f"task={task['task_id']} "
                        f"xy={metrics.get('final_xy', 999):.4f} "
                        f"lift={metrics.get('lift_delta', 0):.4f}"
                    )

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
