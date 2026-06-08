import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import robosuite as suite
from robosuite import load_composite_controller_config


def try_make_env(env_name):
    print()
    print("=" * 80)
    print("Trying env:", env_name)

    try:
        controller_config = load_composite_controller_config(
            controller="BASIC",
            robot="Panda",
        )

        env = suite.make(
            env_name=env_name,
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["agentview", "frontview"],
            camera_heights=128,
            camera_widths=128,
            reward_shaping=True,
            control_freq=20,
            horizon=200,
        )

        obs = env.reset()

        print("SUCCESS:", env_name)
        print("Action dim:", env.action_dim)

        print("Obs keys:")
        for k in sorted(obs.keys()):
            if "image" in k or "pos" in k or "quat" in k or "object" in k:
                try:
                    shape = obs[k].shape
                except Exception:
                    shape = type(obs[k])
                print(" ", k, shape)

        print("Objects:")
        try:
            print([getattr(o, "name", str(o)) for o in env.objects])
        except Exception as e:
            print("No env.objects:", repr(e))

        print("Body names:")
        try:
            names = []
            for i in range(env.sim.model.nbody):
                name = env.sim.model.body_id2name(i)
                if name is not None:
                    names.append(name)

            for name in names[:100]:
                print(" ", name)
        except Exception as e:
            print("Could not list body names:", repr(e))

        env.close()
        return True

    except Exception as e:
        print("FAILED:", env_name)
        print("Reason:", repr(e))
        return False


def main():
    print("robosuite version:", getattr(suite, "__version__", "unknown"))
    print("robosuite module:", suite.__file__)

    candidates = [
        "Lift",
        "Stack",
        "PickPlace",
        "PickPlaceCan",
        "PickPlaceMilk",
        "PickPlaceBread",
        "PickPlaceCereal",
        "NutAssembly",
        "Door",
        "ToolHang",
    ]

    working = []

    for env_name in candidates:
        ok = try_make_env(env_name)
        if ok:
            working.append(env_name)

    print()
    print("=" * 80)
    print("Working envs:")
    for env_name in working:
        print(" ", env_name)


if __name__ == "__main__":
    main()
