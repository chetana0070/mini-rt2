import os
import itertools
import numpy as np

from scripts import __path__  # keeps script import-safe

from src.living_room.pickplace_env import make_pickplace_env
from src.living_room.place_controller import scripted_place_policy


TASK_OBJECT = "Milk"
BIN_POS = np.array([0.1, -0.25, 0.8], dtype=np.float32)


def run_once(grasp_z, close_peak, hold_cmd, seed_i):
    os.environ["ROBOT_NAME"] = "UR5e"
    os.environ["UR5E_OPEN"] = "-0.20"
    os.environ["UR5E_GRASP_Z_ADD"] = str(grasp_z)
    os.environ["UR5E_PREGRASP_Z_ADD"] = "0.145"
    os.environ["UR5E_CLOSE_PEAK"] = str(close_peak)
    os.environ["UR5E_HOLD"] = str(hold_cmd)
    os.environ["UR5E_GRASP_BIAS_X"] = "0.0"
    os.environ["UR5E_GRASP_BIAS_Y"] = "0.0"
    os.environ["UR5E_PLACE_BIAS_X"] = "0.0"

    env, _ = make_pickplace_env(
        render=False,
        robot="UR5e",
        camera_height=64,
        camera_width=64,
        horizon=430,
    )

    obs = env.reset()
    obj_start = obs[f"{TASK_OBJECT}_pos"].copy()

    max_lift = 0.0
    max_xy_jump = 0.0
    bad = False
    last_stage = ""

    for step in range(430):
        action, stage = scripted_place_policy(
            obs=obs,
            step=step,
            env=env,
            obj_name=TASK_OBJECT,
            obj_start=obj_start,
            bin_pos=BIN_POS,
        )
        obs, reward, done, info = env.step(action)

        obj = obs[f"{TASK_OBJECT}_pos"]
        lift = float(obj[2] - obj_start[2])
        max_lift = max(max_lift, lift)
        max_xy_jump = max(max_xy_jump, float(np.linalg.norm(obj[:2] - obj_start[:2])))
        last_stage = stage

        if stage == "abort_object_unstable" or np.linalg.norm(obj[:2]) > 1.25 or obj[2] < 0.05:
            bad = True
            break

    env.close()

    ok = (not bad) and max_lift > 0.08 and max_xy_jump < 0.35
    return {
        "ok": ok,
        "bad": bad,
        "max_lift": max_lift,
        "max_xy_jump": max_xy_jump,
        "last_stage": last_stage,
        "grasp_z": grasp_z,
        "close_peak": close_peak,
        "hold": hold_cmd,
    }


def main():
    grasp_z_values = [0.020, 0.035, 0.050, 0.065, 0.080]
    close_values = [0.008, 0.012, 0.018, 0.025, 0.030]
    hold_values = [0.000, 0.004, 0.006, 0.008, 0.010]

    results = []

    for i, (gz, cp, hold) in enumerate(itertools.product(grasp_z_values, close_values, hold_values)):
        r = run_once(gz, cp, hold, i)
        results.append(r)
        print(
            f"gz={gz:.3f} close={cp:.3f} hold={hold:.3f} "
            f"ok={r['ok']} bad={r['bad']} "
            f"lift={r['max_lift']:.3f} xyjump={r['max_xy_jump']:.3f} "
            f"stage={r['last_stage']}"
        )

    good = [r for r in results if r["ok"]]
    print("\n=== GOOD RESULTS ===")
    for r in good[:20]:
        print(r)

    if not good:
        print("\nNo stable pickup found. Next fix is grasp orientation / side approach, not more strength.")


if __name__ == "__main__":
    main()
