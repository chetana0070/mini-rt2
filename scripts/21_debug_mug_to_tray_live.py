import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--horizon", type=int, default=260)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
args = parser.parse_args()

os.environ["MUJOCO_GL"] = args.gl

import imageio
import numpy as np

from src.envs.render_utils import show_camera_panel, close_render_windows
from src.living_room.pickplace_env import make_pickplace_env, get_bin_pos


TARGET_OBJECT = "Milk"
TASK_ID = "mug_to_tray"
INSTRUCTION = "put the mug in the tray"


def move_to(obs, target, gripper, action_dim, gain=8.0, max_step=0.8):
    eef = obs["robot0_eef_pos"]
    delta = target - eef

    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = np.clip(delta * gain, -max_step, max_step)

    if action_dim >= 6:
        action[3:6] = 0.0

    action[-1] = gripper
    return action


def scripted_policy(obs, step, env, obj_start, bin_pos):
    open_gripper = -1.0
    close_gripper = 1.0

    obj = obs[f"{TARGET_OBJECT}_pos"].copy()

    # PickPlace objects are taller than cube.
    above_obj = np.array([obj[0], obj[1], obj_start[2] + 0.18])
    pregrasp = np.array([obj[0], obj[1], obj_start[2] + 0.07])
    grasp = np.array([obj[0], obj[1], obj_start[2] + 0.015])

    carry = np.array([obj[0], obj[1], obj_start[2] + 0.28])

    # bin1 body center may be low; use safe tray drop height.
    above_bin = np.array([bin_pos[0], bin_pos[1], obj_start[2] + 0.30])
    drop = np.array([bin_pos[0], bin_pos[1], obj_start[2] + 0.12])

    retreat = np.array([bin_pos[0], bin_pos[1], obj_start[2] + 0.30])

    if step < 40:
        stage = "reach_above_mug"
        action = move_to(obs, above_obj, open_gripper, env.action_dim, gain=9.0)

    elif step < 80:
        stage = "pregrasp_mug"
        action = move_to(obs, pregrasp, open_gripper, env.action_dim, gain=8.0)

    elif step < 115:
        stage = "descend_to_mug"
        action = move_to(obs, grasp, open_gripper, env.action_dim, gain=7.0)

    elif step < 150:
        stage = "close_gripper"
        action = move_to(obs, grasp, close_gripper, env.action_dim, gain=5.0)

    elif step < 180:
        stage = "lift_mug"
        action = move_to(obs, carry, close_gripper, env.action_dim, gain=6.0)

    elif step < 215:
        stage = "move_to_tray"
        action = move_to(obs, above_bin, close_gripper, env.action_dim, gain=5.0)

    elif step < 235:
        stage = "lower_to_tray"
        action = move_to(obs, drop, close_gripper, env.action_dim, gain=5.0)

    elif step < 250:
        stage = "release"
        action = move_to(obs, drop, open_gripper, env.action_dim, gain=4.0)

    else:
        stage = "retreat"
        action = move_to(obs, retreat, open_gripper, env.action_dim, gain=5.0)

    return action, stage


def success_check(obs, bin_pos):
    obj = obs[f"{TARGET_OBJECT}_pos"]
    xy_dist = np.linalg.norm(obj[:2] - bin_pos[:2])
    z_ok = obj[2] > 0.80
    return bool(xy_dist < 0.12 and z_ok), xy_dist


def main():
    os.makedirs("videos/living_room", exist_ok=True)

    env, camera_names = make_pickplace_env(
        render=True,
        camera_height=args.height,
        camera_width=args.width,
        horizon=args.horizon,
    )

    obs = env.reset()

    camera_keys = [f"{name}_image" for name in camera_names]
    obj_start = obs[f"{TARGET_OBJECT}_pos"].copy()
    bin_pos = get_bin_pos(env, "bin1")

    print("Task:", TASK_ID)
    print("Instruction:", INSTRUCTION)
    print("Object:", TARGET_OBJECT, "alias=mug")
    print("Action dim:", env.action_dim)
    print("Object start:", np.round(obj_start, 3))
    print("bin1 pos:", np.round(bin_pos, 3))
    print("GL:", args.gl)
    print("Live rendering ON.")
    print("Press q in camera window to stop.")

    frames = []
    success = False
    best_xy = 999.0

    for step in range(args.horizon):
        action, stage = scripted_policy(
            obs=obs,
            step=step,
            env=env,
            obj_start=obj_start,
            bin_pos=bin_pos,
        )

        obs, reward, done, info = env.step(action)

        ok, xy_dist = success_check(obs, bin_pos)
        best_xy = min(best_xy, xy_dist)

        env.render()

        obj_pos = obs[f"{TARGET_OBJECT}_pos"]

        title = (
            f"{TASK_ID} step={step} stage={stage} "
            f"xy_to_bin={xy_dist:.3f} obj_z={obj_pos[2]:.3f}"
        )

        stop = show_camera_panel(
            obs=obs,
            camera_keys=camera_keys,
            window_name="Mini-RT2-Room: Mug to Tray Debug",
            title_text=title,
            delay=1,
        )

        frames.append(obs["agentview_image"][::-1])

        if step % 20 == 0:
            print(
                f"step={step:03d} stage={stage:<16} "
                f"obj={np.round(obj_pos, 3)} "
                f"eef={np.round(obs['robot0_eef_pos'], 3)} "
                f"xy_to_bin={xy_dist:.3f} reward={reward:.3f}"
            )

        if ok and step > 230:
            success = True
            print(f"SUCCESS at step={step}, xy_to_bin={xy_dist:.3f}")
            break

        if done or stop:
            print("Stopped.")
            break

        time.sleep(0.01)

    video_path = "videos/living_room/debug_mug_to_tray_live.mp4"
    imageio.mimsave(video_path, frames, fps=20)

    env.close()
    close_render_windows()

    print()
    print("Done.")
    print("success:", success)
    print("best xy_to_bin:", round(float(best_xy), 4))
    print("video:", video_path)


if __name__ == "__main__":
    main()
