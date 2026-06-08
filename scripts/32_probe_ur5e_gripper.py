import argparse
import os
import time
import numpy as np

from src.living_room.pickplace_env import make_pickplace_env


def get_gripper_obs(obs):
    keys = [k for k in obs.keys() if "gripper" in k.lower()]
    out = {}
    for k in keys:
        try:
            out[k] = np.asarray(obs[k]).round(5)
        except Exception:
            out[k] = obs[k]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="UR5e", choices=["UR5e", "Panda"])
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")

    env, camera_names = make_pickplace_env(
        render=args.render,
        robot=args.robot,
        camera_height=args.height,
        camera_width=args.width,
        horizon=1000,
    )

    obs = env.reset()

    low, high = env.action_spec
    print("robot:", args.robot)
    print("action_dim:", env.action_dim)
    print("action_low:", low)
    print("action_high:", high)
    print("gripper obs keys:", [k for k in obs.keys() if "gripper" in k.lower()])
    print("initial gripper obs:", get_gripper_obs(obs))

    # Test commands without moving arm.
    # If robot is stable, only fingers should move.
    commands = [-1.0, 1.0, -1.0, 0.0, 0.35, 0.50, 0.70, 1.0, -1.0]

    try:
        for cmd in commands:
            print("\n=== gripper command:", cmd, "===")

            for t in range(args.steps):
                action = np.zeros(env.action_dim, dtype=np.float32)
                action[-1] = cmd
                obs, reward, done, info = env.step(action)

                if args.render:
                    env.render()

                if t in [0, 10, 30, args.steps - 1]:
                    eef = obs.get("robot0_eef_pos", None)
                    if eef is not None:
                        eef = np.asarray(eef).round(4)
                    print(
                        f"t={t:03d}",
                        "eef=", eef,
                        "gripper=", get_gripper_obs(obs),
                    )

                if done:
                    print("env done, resetting")
                    obs = env.reset()

                time.sleep(0.01)

    finally:
        env.close()


if __name__ == "__main__":
    main()
