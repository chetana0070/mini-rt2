import os

# For live window rendering, GLFW is better than EGL.
# If GLFW fails on your machine, comment this line and use egl.
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
import numpy as np
import imageio

sys.path.append(os.path.abspath("."))

from src.envs.panda_lift_env import make_panda_lift_env
from src.envs.render_utils import show_camera_panel, close_render_windows


def main():
    env, camera_names = make_panda_lift_env(
        render=True,
        camera_height=128,
        camera_width=128,
        horizon=200,
    )

    obs = env.reset()

    camera_keys = [f"{name}_image" for name in camera_names]

    print("Environment created.")
    print("Action dim:", env.action_dim)
    print("Camera keys requested:", camera_keys)
    print("Available obs image keys:", [k for k in obs.keys() if "image" in k])
    print("Press q in the camera window to stop.")

    frames = []
    total_reward = 0.0

    for step in range(200):
        # Random actions for visual test only
        action = np.random.uniform(-0.2, 0.2, env.action_dim)

        obs, reward, done, info = env.step(action)

        # Whole-scene robosuite viewer
        env.render()

        # Side-by-side camera window
        stop = show_camera_panel(
            obs,
            camera_keys=camera_keys,
            window_name="Mini RT-2: agentview + whole scene",
            delay=1,
        )

        if "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])

        total_reward += reward

        time.sleep(0.02)

        if done or stop:
            break

    env.close()
    close_render_windows()

    os.makedirs("videos", exist_ok=True)
    out_path = "videos/panda_lift_rendered_test.mp4"

    if len(frames) > 0:
        imageio.mimsave(out_path, frames, fps=20)
        print("Saved video:", out_path)

    print("Simulation worked.")
    print("Steps:", len(frames))
    print("Total reward:", round(float(total_reward), 4))


if __name__ == "__main__":
    main()
