import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="PickPlace")
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
args = parser.parse_args()

os.environ["MUJOCO_GL"] = args.gl

import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config

from src.envs.render_utils import show_camera_panel, close_render_windows


def make_env():
    controller_config = load_composite_controller_config(
        controller="BASIC",
        robot="Panda",
    )

    env = suite.make(
        env_name=args.env,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "frontview"],
        camera_heights=args.height,
        camera_widths=args.width,
        reward_shaping=True,
        control_freq=20,
        horizon=args.steps,
    )

    return env


def print_obs(obs):
    print("Obs keys:")
    for k in sorted(obs.keys()):
        if "image" in k or "pos" in k or "quat" in k or "object" in k:
            try:
                print(" ", k, obs[k].shape)
            except Exception:
                print(" ", k, type(obs[k]))

    print()
    print("Object aliases:")
    aliases = {
        "Milk": "mug",
        "Bread": "book",
        "Cereal": "box",
        "Can": "remote",
    }

    for real, alias in aliases.items():
        key = f"{real}_pos"
        if key in obs:
            print(f" {real:>6} -> {alias:<8} pos={np.round(obs[key], 3)}")


def main():
    env = make_env()
    obs = env.reset()

    print("Live PickPlace viewer")
    print("GL:", args.gl)
    print("Action dim:", env.action_dim)
    print("Press q in camera window to quit.")
    print_obs(obs)

    camera_keys = ["agentview_image", "frontview_image"]

    for step in range(args.steps):
        action = np.zeros(env.action_dim, dtype=np.float32)

        obs, reward, done, info = env.step(action)

        env.render()

        title = (
            f"{args.env} live | step={step} | "
            f"Milk=mug Bread=book Cereal=box Can=remote"
        )

        stop = show_camera_panel(
            obs=obs,
            camera_keys=camera_keys,
            window_name="Mini-RT2-Room Live Cameras",
            title_text=title,
            delay=1,
        )

        if step % 50 == 0:
            print()
            print("step:", step)
            for obj in ["Milk", "Bread", "Cereal", "Can"]:
                key = f"{obj}_pos"
                if key in obs:
                    print(f" {obj}_pos:", np.round(obs[key], 3))

        if done or stop:
            break

        time.sleep(0.01)

    env.close()
    close_render_windows()
    print("Done.")


if __name__ == "__main__":
    main()
