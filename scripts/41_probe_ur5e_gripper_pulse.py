import os
import time
import numpy as np
import robosuite as suite

os.environ.setdefault("MUJOCO_GL", "glfw")


def gripper_qpos(obs):
    return np.round(obs["robot0_gripper_qpos"], 4)


def run(close_steps, hold_cmd, label):
    env = suite.make(
        env_name="PickPlace",
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        single_object_mode=2,
        object_type="milk",
        horizon=300,
    )

    obs = env.reset()
    print(f"\n=== {label} close_steps={close_steps} hold_cmd={hold_cmd} ===")
    print("start qpos:", gripper_qpos(obs))

    for t in range(160):
        action = np.zeros(env.action_dim, dtype=np.float32)

        if t < 40:
            action[-1] = -1.0       # open first
        elif t < 40 + close_steps:
            action[-1] = 1.0        # full close pulse
        else:
            action[-1] = hold_cmd   # test hold

        obs, _, _, _ = env.step(action)
        env.render()

        if t in [0, 39, 45, 55, 70, 100, 140, 159]:
            print(f"t={t:03d} cmd={action[-1]:.2f} qpos={gripper_qpos(obs)}")

        time.sleep(0.005)

    env.close()


def main():
    tests = [
        (5, 0.0, "pulse5_hold0"),
        (10, 0.0, "pulse10_hold0"),
        (20, 0.0, "pulse20_hold0"),
        (5, -0.05, "pulse5_tinyopen"),
        (10, -0.05, "pulse10_tinyopen"),
    ]

    for close_steps, hold_cmd, label in tests:
        run(close_steps, hold_cmd, label)


if __name__ == "__main__":
    main()
