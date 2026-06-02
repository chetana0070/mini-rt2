import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import imageio
import robosuite as suite


def make_env():
    try:
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(
            controller="BASIC",
            robot="Panda",
        )

        env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["agentview"],
            camera_heights=128,
            camera_widths=128,
            control_freq=20,
            horizon=200,
            reward_shaping=True,
        )
        return env

    except Exception as e:
        print("Composite controller failed. Trying old controller API.")
        print("Reason:", repr(e))

        controller_config = suite.load_controller_config(default_controller="OSC_POSE")

        env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["agentview"],
            camera_heights=128,
            camera_widths=128,
            control_freq=20,
            horizon=200,
            reward_shaping=True,
        )
        return env


def main():
    env = make_env()
    obs = env.reset()

    print("Environment created.")
    print("Action dim:", env.action_dim)
    print("Observation keys:", list(obs.keys()))

    frames = []
    total_reward = 0.0

    for step in range(100):
        action = np.random.uniform(-0.3, 0.3, env.action_dim)
        obs, reward, done, info = env.step(action)

        frame = obs["agentview_image"]
        frames.append(frame)

        total_reward += reward

        if done:
            break

    env.close()

    os.makedirs("videos", exist_ok=True)
    out_path = "videos/panda_lift_random.mp4"
    imageio.mimsave(out_path, frames, fps=20)

    print("Simulation worked.")
    print("Steps:", len(frames))
    print("Total reward:", round(float(total_reward), 4))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
