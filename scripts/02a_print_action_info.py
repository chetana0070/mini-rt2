import os
os.environ["MUJOCO_GL"] = "egl"

import sys
sys.path.append(os.path.abspath("."))

from src.envs.panda_lift_env import make_panda_lift_env

env, camera_names = make_panda_lift_env(render=False)

print("Action dim:", env.action_dim)
print("Camera names:", camera_names)
print()

try:
    env.robots[0].print_action_info()
except Exception as e:
    print("Could not print action info:", repr(e))

env.close()
