import math
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

os.environ["MUJOCO_GL"] = "glfw"

import numpy as np

from src.envs.render_utils import show_camera_panel, close_render_windows
from src.living_room.pickplace_env import make_pickplace_env


OBJ = "Cereal"


def move_to(obs, target, gripper, action_dim, gain=5.0, max_step=0.55, yaw_cmd=0.0):
    eef = obs["robot0_eef_pos"]
    delta = target - eef

    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = np.clip(delta * gain, -max_step, max_step)

    # Try orientation command.
    # In this controller, action[5] usually affects wrist yaw / z-rotation.
    if action_dim >= 6:
        action[3] = 0.0
        action[4] = 0.0
        action[5] = yaw_cmd

    action[-1] = gripper
    return action


def run_trial(yaw_cmd, grasp_offset, trial_id):
    env, camera_names = make_pickplace_env(
        render=True,
        camera_height=128,
        camera_width=128,
        horizon=380,
    )

    obs = env.reset()
    camera_keys = [f"{name}_image" for name in camera_names]

    start = obs[f"{OBJ}_pos"].copy()
    z0 = float(start[2])

    xy = start[:2].copy()

    safe_z = max(z0 + 0.40, 1.20)
    above = np.array([xy[0], xy[1], safe_z])
    pre = np.array([xy[0], xy[1], z0 + 0.150])
    grasp = np.array([xy[0], xy[1], z0 + grasp_offset])
    lift = np.array([xy[0], xy[1], safe_z])

    max_lift = 0.0

    print()
    print("=" * 80)
    print(f"trial={trial_id} yaw_cmd={yaw_cmd:.2f} grasp_offset={grasp_offset:.3f}")
    print("start:", np.round(start, 3))

    for step in range(380):
        if step < 80:
            stage = "above"
            action = move_to(obs, above, -1.0, env.action_dim, gain=6.0, yaw_cmd=yaw_cmd)
        elif step < 160:
            stage = "pregrasp"
            action = move_to(obs, pre, -1.0, env.action_dim, gain=5.0, yaw_cmd=yaw_cmd)
        elif step < 245:
            stage = "descend"
            action = move_to(obs, grasp, -1.0, env.action_dim, gain=3.8, yaw_cmd=yaw_cmd)
        elif step < 310:
            stage = "close"
            action = move_to(obs, grasp, 1.0, env.action_dim, gain=3.2, yaw_cmd=yaw_cmd)
        else:
            stage = "lift"
            action = move_to(obs, lift, 1.0, env.action_dim, gain=4.5, yaw_cmd=yaw_cmd)

        obs, reward, done, info = env.step(action)

        obj = obs[f"{OBJ}_pos"]
        max_lift = max(max_lift, float(obj[2] - z0))

        env.render()

        title = (
            f"toy yaw sweep trial={trial_id} stage={stage} "
            f"yaw={yaw_cmd:.2f} gz={grasp_offset:.3f} lift={max_lift:.3f}"
        )

        stop = show_camera_panel(
            obs=obs,
            camera_keys=camera_keys,
            window_name="Toy Yaw Grasp Sweep",
            title_text=title,
            delay=1,
        )

        if step % 30 == 0:
            print(
                f"step={step:03d} stage={stage:<8} "
                f"obj={np.round(obj, 3)} "
                f"eef={np.round(obs['robot0_eef_pos'], 3)} "
                f"lift={max_lift:.3f}"
            )

        if max_lift >= 0.10 and step > 330:
            break

        if done or stop:
            break

        time.sleep(0.005)

    env.close()
    close_render_windows()

    success = max_lift >= 0.10
    print("trial result:", "SUCCESS" if success else "FAIL", "max_lift=", round(max_lift, 4))

    return success, max_lift


def main():
    yaw_cmds = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    grasp_offsets = [0.035, 0.045, 0.055, 0.065]

    results = []
    trial_id = 0

    for yaw in yaw_cmds:
        for gz in grasp_offsets:
            success, max_lift = run_trial(yaw, gz, trial_id)
            results.append((success, max_lift, yaw, gz))
            trial_id += 1

            if success:
                print()
                print("FOUND WORKING YAW GRASP")
                print("yaw_cmd:", yaw)
                print("grasp_offset:", gz)
                print("max_lift:", round(max_lift, 4))
                print()
                print("Top results:")
                for item in sorted(results, key=lambda x: x[1], reverse=True)[:10]:
                    print(item)
                return

    print()
    print("No robust yaw grasp found.")
    print("Top results:")
    for item in sorted(results, key=lambda x: x[1], reverse=True)[:10]:
        print(item)


if __name__ == "__main__":
    main()
