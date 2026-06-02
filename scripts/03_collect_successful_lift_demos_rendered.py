import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--target_successes", type=int, default=10)
parser.add_argument("--max_attempts", type=int, default=20)
parser.add_argument("--horizon", type=int, default=260)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--out", type=str, default="datasets/lift_success_demos.hdf5")
parser.add_argument("--no_render", action="store_true")
args = parser.parse_args()

os.environ["MUJOCO_GL"] = args.gl

import h5py
import imageio
import numpy as np

sys.path.append(os.path.abspath("."))

from src.envs.panda_lift_env import make_panda_lift_env
from src.envs.render_utils import show_camera_panel, close_render_windows


BEST_GRASP_OFFSET = -0.010
SUCCESS_LIFT_DELTA = 0.08
INSTRUCTION = "lift the cube"


def move_to(obs, target, gripper, action_dim, gain=12.0, max_step=0.8):
    eef = obs["robot0_eef_pos"]
    delta = target - eef

    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = np.clip(delta * gain, -max_step, max_step)

    if action_dim >= 6:
        action[3:6] = 0.0

    action[-1] = gripper
    return action


def scripted_lift_policy(obs, step, action_dim, cube_start):
    close_sign = 1.0
    open_sign = -1.0

    cube = obs["cube_pos"].copy()

    above = np.array([cube[0], cube[1], cube_start[2] + 0.18])
    pregrasp = np.array([cube[0], cube[1], cube_start[2] + 0.06])
    grasp = np.array([cube[0], cube[1], cube_start[2] + BEST_GRASP_OFFSET])
    lift = np.array([cube_start[0], cube_start[1], cube_start[2] + 0.32])

    if step < 45:
        stage = "above"
        action = move_to(obs, above, open_sign, action_dim, gain=10.0)

    elif step < 85:
        stage = "pregrasp"
        action = move_to(obs, pregrasp, open_sign, action_dim, gain=10.0)

    elif step < 125:
        stage = "descend"
        action = move_to(obs, grasp, open_sign, action_dim, gain=8.0)

    elif step < 175:
        stage = "close_hold"
        action = move_to(obs, grasp, close_sign, action_dim, gain=6.0)

    else:
        stage = "slow_lift"
        action = move_to(obs, lift, close_sign, action_dim, gain=5.0)

    return action, stage


def main():
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    render = not args.no_render

    env, camera_names = make_panda_lift_env(
        render=render,
        camera_height=args.height,
        camera_width=args.width,
        horizon=args.horizon,
    )

    camera_keys = [f"{name}_image" for name in camera_names]

    print("Collecting successful Mini RT-2 Lift demos")
    print("Target successes:", args.target_successes)
    print("Max attempts:", args.max_attempts)
    print("Output:", args.out)
    print("GL:", args.gl)
    print("Render:", render)
    print("Action dim:", env.action_dim)
    print("Camera keys:", camera_keys)
    print("Best grasp offset:", BEST_GRASP_OFFSET)
    print("Press q in camera window to stop current episode early.")

    success_count = 0
    attempt_count = 0

    with h5py.File(args.out, "w") as f:
        f.attrs["env"] = "robosuite Lift"
        f.attrs["robot"] = "Panda"
        f.attrs["instruction"] = INSTRUCTION
        f.attrs["action_dim"] = env.action_dim
        f.attrs["camera_names"] = ",".join(camera_names)
        f.attrs["gripper_close"] = 1.0
        f.attrs["gripper_open"] = -1.0
        f.attrs["best_grasp_offset"] = BEST_GRASP_OFFSET
        f.attrs["success_lift_delta"] = SUCCESS_LIFT_DELTA

        while success_count < args.target_successes and attempt_count < args.max_attempts:
            obs = env.reset()

            cube_start = obs["cube_pos"].copy()
            start_cube_z = float(cube_start[2])
            max_cube_z = start_cube_z

            ep_agent = []
            ep_front = []
            ep_actions = []
            ep_eef = []
            ep_cube = []
            ep_rewards = []
            ep_dones = []
            ep_stages = []
            ep_lift_delta = []

            video_frames = []
            success = False

            for step in range(args.horizon):
                action, stage = scripted_lift_policy(
                    obs=obs,
                    step=step,
                    action_dim=env.action_dim,
                    cube_start=cube_start,
                )

                obs, reward, done, info = env.step(action)

                cube_z = float(obs["cube_pos"][2])
                max_cube_z = max(max_cube_z, cube_z)
                lift_delta = max_cube_z - start_cube_z

                if render:
                    env.render()

                title = (
                    f"attempt={attempt_count} success={success_count}/{args.target_successes} "
                    f"step={step} {stage} dz={lift_delta:.3f}"
                )

                stop = show_camera_panel(
                    obs=obs,
                    camera_keys=camera_keys,
                    window_name="Mini RT-2 Successful Demo Collection",
                    title_text=title,
                    delay=1,
                )

                ep_agent.append(obs["agentview_image"])
                ep_front.append(obs["frontview_image"])
                ep_actions.append(action)
                ep_eef.append(obs["robot0_eef_pos"])
                ep_cube.append(obs["cube_pos"])
                ep_rewards.append(reward)
                ep_dones.append(done)
                ep_stages.append(stage.encode("utf-8"))
                ep_lift_delta.append(lift_delta)

                video_frames.append(obs["agentview_image"][::-1])

                if lift_delta >= SUCCESS_LIFT_DELTA:
                    success = True
                    print(
                        f"Attempt {attempt_count}: SUCCESS "
                        f"step={step}, lift_delta={lift_delta:.4f}"
                    )
                    break

                if done or stop:
                    print(
                        f"Attempt {attempt_count}: stopped "
                        f"step={step}, lift_delta={lift_delta:.4f}"
                    )
                    break

                time.sleep(0.005)

            if success:
                demo_name = f"demo_{success_count:04d}"
                g = f.create_group(demo_name)

                g.create_dataset("agentview_image", data=np.asarray(ep_agent, dtype=np.uint8), compression="gzip")
                g.create_dataset("frontview_image", data=np.asarray(ep_front, dtype=np.uint8), compression="gzip")
                g.create_dataset("actions", data=np.asarray(ep_actions, dtype=np.float32))
                g.create_dataset("eef_pos", data=np.asarray(ep_eef, dtype=np.float32))
                g.create_dataset("cube_pos", data=np.asarray(ep_cube, dtype=np.float32))
                g.create_dataset("rewards", data=np.asarray(ep_rewards, dtype=np.float32))
                g.create_dataset("dones", data=np.asarray(ep_dones, dtype=np.bool_))
                g.create_dataset("stages", data=np.asarray(ep_stages))
                g.create_dataset("lift_delta", data=np.asarray(ep_lift_delta, dtype=np.float32))

                g.attrs["instruction"] = INSTRUCTION
                g.attrs["success"] = True
                g.attrs["attempt"] = attempt_count
                g.attrs["final_lift_delta"] = float(ep_lift_delta[-1])

                video_path = f"videos/success_lift_demo_{success_count:04d}.mp4"
                imageio.mimsave(video_path, video_frames, fps=20)

                print(f"Saved {demo_name}: steps={len(ep_actions)}, video={video_path}")

                success_count += 1
            else:
                print(f"Attempt {attempt_count}: failed, not saved")

            attempt_count += 1

        f.attrs["successful_episodes"] = success_count
        f.attrs["attempts"] = attempt_count

    env.close()
    close_render_windows()

    print("Done.")
    print("Successes:", success_count)
    print("Attempts:", attempt_count)
    print("Dataset:", args.out)


if __name__ == "__main__":
    main()
