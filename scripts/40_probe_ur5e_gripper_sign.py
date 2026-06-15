import os
import time
import numpy as np
import robosuite as suite

os.environ.setdefault("MUJOCO_GL", "glfw")


def print_gripper_info(env, obs, label):
    print("\n===", label, "===")

    print("obs keys with gripper:")
    for k in obs.keys():
        if "gripper" in k.lower():
            print(k, np.round(obs[k], 5))

    print("joint names containing gripper/finger:")
    for i in range(env.sim.model.njnt):
        name = env.sim.model.joint_id2name(i)
        if name and any(x in name.lower() for x in ["gripper", "finger", "robotiq"]):
            qadr = env.sim.model.jnt_qposadr[i]
            print(i, name, "qpos=", round(float(env.sim.data.qpos[qadr]), 5))

    print("actuator names containing gripper/finger:")
    for i in range(env.sim.model.nu):
        name = env.sim.model.actuator_id2name(i)
        if name and any(x in name.lower() for x in ["gripper", "finger", "robotiq"]):
            print(i, name, "ctrl=", round(float(env.sim.data.ctrl[i]), 5))


def run_cmd(cmd, label):
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
    print_gripper_info(env, obs, label + " start")

    for t in range(120):
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[-1] = cmd
        obs, _, _, _ = env.step(action)
        env.render()
        time.sleep(0.005)

    print_gripper_info(env, obs, label + f" after cmd={cmd}")

    env.close()


def main():
    print("Testing UR5e gripper sign.")
    print("Watch visually too.")
    print("cmd=-1, -0.5, +0.5, +1")

    for cmd in [-1.0, -0.5, 0.5, 1.0]:
        run_cmd(cmd, f"test_{cmd}")


if __name__ == "__main__":
    main()
