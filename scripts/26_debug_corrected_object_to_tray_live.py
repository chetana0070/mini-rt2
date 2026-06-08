import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="remote_to_right_tray")
parser.add_argument("--robot", type=str, default="UR5e", choices=["UR5e", "Panda"])
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--horizon", type=int, default=720)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
args = parser.parse_args()

os.environ["MUJOCO_GL"] = args.gl

import imageio
import numpy as np

from src.envs.render_utils import show_camera_panel, close_render_windows
from src.living_room.pickplace_env import make_pickplace_env, get_bin_pos
from src.living_room.place_controller import scripted_place_policy, strict_success


def load_task(task_id):
    cfg = json.loads(Path("configs/living_room/tasks_v0.json").read_text())
    for task in cfg["tasks"]:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Task not found: {task_id}")


def main():
    Path("videos/living_room").mkdir(parents=True, exist_ok=True)

    task = load_task(args.task)

    obj_name = task["robosuite_object"]
    instruction = task["instruction"]

    env, camera_names = make_pickplace_env(
        render=True,
        robot=args.robot,
        camera_height=args.height,
        camera_width=args.width,
        horizon=args.horizon,
    )

    obs = env.reset()

    camera_keys = [f"{name}_image" for name in camera_names]

    obj_start = obs[f"{obj_name}_pos"].copy()
    bin_pos = get_bin_pos(env, task["target_container"])

    print("Task:", args.task)
    print("Robot:", args.robot)
    os.environ["ROBOT_NAME"] = args.robot
    print("Instruction:", instruction)
    print("Object:", obj_name)
    print("Target:", task["target_container"])
    print("Object start:", np.round(obj_start, 3))
    print("Bin pos:", np.round(bin_pos, 3))
    print("Live rendering ON")

    frames = []
    max_lift_delta = 0.0
    success = False
    final_metrics = {}

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

        obj_pos = obs[f"{obj_name}_pos"]
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

        env.render()

        title = (
            f"{args.task} step={step} stage={stage} "
            f"xy={metrics['final_xy']:.3f} z={metrics['final_z']:.3f} "
            f"lift={metrics['lift_delta']:.3f}"
        )

        stop = show_camera_panel(
            obs=obs,
            camera_keys=camera_keys,
            window_name="Corrected Placement Debug",
            title_text=title,
            delay=1,
        )

        frames.append(obs["agentview_image"][::-1])

        if step % 20 == 0:
            print(
                f"step={step:03d} stage={stage:<14} "
                f"obj={np.round(obj_pos, 3)} "
                f"eef={np.round(obs['robot0_eef_pos'], 3)} "
                f"xy={metrics['final_xy']:.3f} "
                f"z={metrics['final_z']:.3f} "
                f"lift={metrics['lift_delta']:.3f}"
            )

        if ok:
            success = True
            print("STRICT SUCCESS at step:", step)
            break

        if done or stop:
            break

        time.sleep(0.01)

    video_path = f"videos/living_room/corrected_debug_{args.task}.mp4"
    imageio.mimsave(video_path, frames, fps=20)

    env.close()
    close_render_windows()

    print()
    print("Done.")
    print("success:", success)
    print("final_metrics:", final_metrics)
    print("video:", video_path)


if __name__ == "__main__":
    main()
