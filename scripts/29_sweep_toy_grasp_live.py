import itertools
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


def move_to(obs, target, gripper, action_dim, gain=5.0, max_step=0.60):
    eef = obs["robot0_eef_pos"]
    delta = target - eef

    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = np.clip(delta * gain, -max_step, max_step)
    action[3:6] = 0.0
    action[-1] = gripper
    return action


def run_trial(xy_bias, grasp_offset, trial_id):
    env, camera_names = make_pickplace_env(
        render=True,
        camera_height=128,
        camera_width=128,
        horizon=360,
    )

    obs = env.reset()
    camera_keys = [f"{name}_image" for name in camera_names]

    start = obs[f"{OBJ}_pos"].copy()
    z0 = float(start[2])

    xy = start[:2] + np.array(xy_bias, dtype=np.float32)

    safe_z = max(z0 + 0.40, 1.20)
    above = np.array([xy[0], xy[1], safe_z])
    pre = np.array([xy[0], xy[1], z0 + 0.15])
    grasp = np.array([xy[0], xy[1], z0 + grasp_offset])
    lift = np.array([xy[0], xy[1], safe_z])

    max_lift = 0.0
    stop = False

    print()
    print("=" * 80)
    print(f"trial={trial_id} xy_bias={xy_bias} grasp_offset={grasp_offset}")
    print("start:", np.round(start, 3))

    for step in range(360):
        if step < 70:
            stage = "above"
            action = move_to(obs, above, -1.0, env.action_dim, gain=6.0)
        elif step < 150:
            stage = "pregrasp"
            action = move_to(obs, pre, -1.0, env.action_dim, gain=5.0)
        elif step < 230:
            stage = "descend"
            action = move_to(obs, grasp, -1.0, env.action_dim, gain=3.8)
        elif step < 290:
            stage = "close"
            action = move_to(obs, grasp, 1.0, env.action_dim, gain=3.2)
        else:
            stage = "lift"
            action = move_to(obs, lift, 1.0, env.action_dim, gain=4.5)

        obs, reward, done, info = env.step(action)

        obj = obs[f"{OBJ}_pos"]
        max_lift = max(max_lift, float(obj[2] - z0))

        env.render()

        title = (
            f"toy grasp sweep trial={trial_id} stage={stage} "
            f"bias={xy_bias} gz={grasp_offset:.3f} lift={max_lift:.3f}"
        )

        stop = show_camera_panel(
            obs=obs,
            camera_keys=camera_keys,
            window_name="Toy Grasp Sweep",
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

        if max_lift >= 0.10 and step > 310:
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
    xy_biases = [
        [0.0, 0.0],
        [0.02, 0.0],
        [-0.02, 0.0],
        [0.0, 0.02],
        [0.0, -0.02],
        [0.03, 0.0],
        [-0.03, 0.0],
    ]

    grasp_offsets = [0.035, 0.045, 0.055, 0.065, 0.075, 0.085]

    results = []
    trial_id = 0

    for xy_bias, grasp_offset in itertools.product(xy_biases, grasp_offsets):
        success, max_lift = run_trial(xy_bias, grasp_offset, trial_id)
        results.append((success, max_lift, xy_bias, grasp_offset))

        trial_id += 1

        if success:
            print()
            print("FOUND WORKING GRASP")
            print("xy_bias:", xy_bias)
            print("grasp_offset:", grasp_offset)
            break

    print()
    print("Top results:")
    results = sorted(results, key=lambda x: x[1], reverse=True)
    for success, max_lift, xy_bias, grasp_offset in results[:10]:
        print(success, round(max_lift, 4), xy_bias, grasp_offset)


if __name__ == "__main__":
    main()
