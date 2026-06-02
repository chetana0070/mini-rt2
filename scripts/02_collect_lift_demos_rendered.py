import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--horizon", type=int, default=180)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--render", action="store_true", default=True)
parser.add_argument("--out", type=str, default="datasets/lift_demos_rendered.hdf5")
args = parser.parse_args()

os.environ["MUJOCO_GL"] = args.gl

import h5py
import imageio
import numpy as np

sys.path.append(os.path.abspath("."))

from src.envs.panda_lift_env import make_panda_lift_env
from src.envs.render_utils import show_camera_panel, close_render_windows


def move_action(obs, target_pos, gripper_cmd, action_dim):
    eef_pos = obs["robot0_eef_pos"]
    delta = target_pos - eef_pos

    action = np.zeros(action_dim, dtype=np.float32)

    # Position control part
    action[:3] = np.clip(delta * 8.0, -0.8, 0.8)

    # Rotation part kept still
    if action_dim >= 6:
        action[3:6] = 0.0

    # Gripper command
    action[-1] = gripper_cmd

    return action


def scripted_lift_policy(obs, step, action_dim):
    cube_pos = obs["cube_pos"].copy()
    eef_pos = obs["robot0_eef_pos"].copy()

    above_cube = cube_pos + np.array([0.0, 0.0, 0.13])
    near_cube = cube_pos + np.array([0.0, 0.0, 0.025])
    lift_target = cube_pos + np.array([0.0, 0.0, 0.28])

    # Gripper convention in robosuite Panda:
    # usually +1 closes, -1 opens.
    open_gripper = -1.0
    close_gripper = 1.0

    dist_above = np.linalg.norm(eef_pos - above_cube)
    dist_near = np.linalg.norm(eef_pos - near_cube)

    if step < 45 or dist_above > 0.035:
        stage = "approach_above_cube"
        action = move_action(obs, above_cube, open_gripper, action_dim)

    elif step < 85 or dist_near > 0.025:
        stage = "descend_to_cube"
        action = move_action(obs, near_cube, open_gripper, action_dim)

    elif step < 115:
        stage = "close_gripper"
        action = move_action(obs, near_cube, close_gripper, action_dim)

    else:
        stage = "lift_cube"
        action = move_action(obs, lift_target, close_gripper, action_dim)

    return action, stage


def is_success(env, obs):
    try:
        return bool(env._check_success())
    except Exception:
        cube_z = float(obs["cube_pos"][2])
        return cube_z > 0.95


def main():
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    env, camera_names = make_panda_lift_env(
        render=args.render,
        camera_height=args.height,
        camera_width=args.width,
        horizon=args.horizon,
    )

    camera_keys = [f"{name}_image" for name in camera_names]

    print("Collecting demos")
    print("Output:", args.out)
    print("GL:", args.gl)
    print("Action dim:", env.action_dim)
    print("Camera keys:", camera_keys)
    print("Press q in OpenCV camera window to stop early.")

    with h5py.File(args.out, "w") as f:
        f.attrs["env"] = "robosuite Lift"
        f.attrs["robot"] = "Panda"
        f.attrs["instruction"] = "lift the cube"
        f.attrs["action_dim"] = env.action_dim
        f.attrs["camera_names"] = ",".join(camera_names)

        successful = 0

        for ep in range(args.episodes):
            obs = env.reset()

            ep_agent = []
            ep_front = []
            ep_actions = []
            ep_eef = []
            ep_cube = []
            ep_rewards = []
            ep_dones = []
            ep_stages = []
            video_frames = []

            success = False

            for step in range(args.horizon):
                action, stage = scripted_lift_policy(obs, step, env.action_dim)

                obs, reward, done, info = env.step(action)

                if args.render:
                    env.render()

                title = f"episode={ep} step={step} stage={stage}"
                stop = show_camera_panel(
                    obs=obs,
                    camera_keys=camera_keys,
                    window_name="Mini RT-2 Demo Collection: agentview + frontview",
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

                video_frames.append(obs["agentview_image"][::-1])

                success = is_success(env, obs)

                if success:
                    successful += 1
                    print(f"Episode {ep}: SUCCESS at step {step}")
                    break

                if done or stop:
                    print(f"Episode {ep}: stopped at step {step}")
                    break

                time.sleep(0.01)

            g = f.create_group(f"demo_{ep:04d}")
            g.create_dataset("agentview_image", data=np.asarray(ep_agent, dtype=np.uint8), compression="gzip")
            g.create_dataset("frontview_image", data=np.asarray(ep_front, dtype=np.uint8), compression="gzip")
            g.create_dataset("actions", data=np.asarray(ep_actions, dtype=np.float32))
            g.create_dataset("eef_pos", data=np.asarray(ep_eef, dtype=np.float32))
            g.create_dataset("cube_pos", data=np.asarray(ep_cube, dtype=np.float32))
            g.create_dataset("rewards", data=np.asarray(ep_rewards, dtype=np.float32))
            g.create_dataset("dones", data=np.asarray(ep_dones, dtype=np.bool_))
            g.create_dataset("stages", data=np.asarray(ep_stages))
            g.attrs["instruction"] = "lift the cube"
            g.attrs["success"] = success

            video_path = f"videos/demo_lift_ep_{ep:04d}.mp4"
            imageio.mimsave(video_path, video_frames, fps=20)

            print(f"Episode {ep}: saved {len(ep_actions)} steps, success={success}")

        f.attrs["successful_episodes"] = successful

    env.close()
    close_render_windows()

    print("Done.")
    print("Successful episodes:", successful, "/", args.episodes)
    print("Dataset:", args.out)


if __name__ == "__main__":
    main()
