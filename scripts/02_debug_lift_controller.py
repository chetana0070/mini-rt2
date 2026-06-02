import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
args = parser.parse_args()

os.environ["MUJOCO_GL"] = args.gl

import imageio
import numpy as np

sys.path.append(os.path.abspath("."))

from src.envs.panda_lift_env import make_panda_lift_env
from src.envs.render_utils import show_camera_panel, close_render_windows


def move_to(obs, target, gripper, action_dim, gain=12.0, max_step=1.0):
    eef = obs["robot0_eef_pos"]
    delta = target - eef

    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = np.clip(delta * gain, -max_step, max_step)

    # Keep orientation fixed
    if action_dim >= 6:
        action[3:6] = 0.0

    action[-1] = gripper
    return action


def run_trial(grasp_z_offset):
    close_sign = 1.0
    open_sign = -1.0

    env, camera_names = make_panda_lift_env(
        render=True,
        camera_height=args.height,
        camera_width=args.width,
        horizon=260,
    )

    obs = env.reset()
    camera_keys = [f"{name}_image" for name in camera_names]

    cube_start = obs["cube_pos"].copy()
    start_cube_z = float(cube_start[2])
    max_cube_z = start_cube_z

    frames = []

    for step in range(260):
        cube = obs["cube_pos"].copy()
        eef = obs["robot0_eef_pos"].copy()

        # Use current cube x/y, but stable z targets
        above = np.array([cube[0], cube[1], cube_start[2] + 0.18])
        pregrasp = np.array([cube[0], cube[1], cube_start[2] + 0.06])
        grasp = np.array([cube[0], cube[1], cube_start[2] + grasp_z_offset])
        lift = np.array([cube_start[0], cube_start[1], cube_start[2] + 0.32])

        if step < 45:
            stage = "above"
            target = above
            grip = open_sign
            gain = 10.0
        elif step < 85:
            stage = "pregrasp"
            target = pregrasp
            grip = open_sign
            gain = 10.0
        elif step < 125:
            stage = "descend"
            target = grasp
            grip = open_sign
            gain = 8.0
        elif step < 175:
            stage = "close_hold"
            target = grasp
            grip = close_sign
            gain = 6.0
        else:
            stage = "slow_lift"
            target = lift
            grip = close_sign
            gain = 5.0

        action = move_to(
            obs=obs,
            target=target,
            gripper=grip,
            action_dim=env.action_dim,
            gain=gain,
            max_step=0.8,
        )

        obs, reward, done, info = env.step(action)

        cube_z = float(obs["cube_pos"][2])
        max_cube_z = max(max_cube_z, cube_z)

        env.render()

        title = (
            f"offset={grasp_z_offset:+.3f} step={step} {stage} "
            f"cube_z={cube_z:.3f} dz={max_cube_z - start_cube_z:.3f}"
        )

        stop = show_camera_panel(
            obs=obs,
            camera_keys=camera_keys,
            window_name="Mini RT-2 Lift Debug",
            title_text=title,
            delay=1,
        )

        frames.append(obs["agentview_image"][::-1])

        if step % 20 == 0:
            print(
                f"offset={grasp_z_offset:+.3f} step={step:03d} "
                f"stage={stage:>10} eef={eef.round(3)} "
                f"cube={cube.round(3)} dz={max_cube_z - start_cube_z:.4f}"
            )

        lift_delta = max_cube_z - start_cube_z

        # Real useful lift threshold
        if lift_delta > 0.08:
            print(f"GOOD LIFT offset={grasp_z_offset:+.3f} at step {step}, dz={lift_delta:.4f}")
            break

        if done or stop:
            break

        time.sleep(0.01)

    os.makedirs("videos", exist_ok=True)
    video_path = f"videos/debug_lift_offset_{grasp_z_offset:+.3f}.mp4"
    imageio.mimsave(video_path, frames, fps=20)

    env.close()
    close_render_windows()

    lift_delta = max_cube_z - start_cube_z

    print()
    print("Trial done")
    print("grasp_z_offset:", grasp_z_offset)
    print("start cube z:", round(start_cube_z, 4))
    print("max cube z:", round(max_cube_z, 4))
    print("lift delta:", round(lift_delta, 4))
    print("video:", video_path)
    print()

    return lift_delta


def main():
    print("Testing better grasp heights.")
    print("Need lift delta > 0.08 m.")
    print()

    offsets = [0.010, 0.005, 0.000, -0.005, -0.010]
    results = {}

    for offset in offsets:
        results[offset] = run_trial(offset)

    print("Summary")
    best_offset = max(results, key=results.get)

    for offset, delta in results.items():
        print(f"offset {offset:+.3f}: lift delta {delta:.4f}")

    print()
    print("Best offset:", f"{best_offset:+.3f}")
    print("Best lift delta:", round(results[best_offset], 4))


if __name__ == "__main__":
    main()
