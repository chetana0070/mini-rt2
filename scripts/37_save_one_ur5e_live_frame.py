import os
import cv2
import numpy as np
import robosuite as suite

os.environ.setdefault("MUJOCO_GL", "glfw")

env = suite.make(
    env_name="PickPlace",
    robots="UR5e",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview"],
    camera_heights=256,
    camera_widths=256,
    single_object_mode=2,
    object_type="milk",
    horizon=10,
)

obs = env.reset()
for _ in range(2):
    obs, _, _, _ = env.step(np.zeros(env.action_dim, dtype=np.float32))

img = np.asarray(obs["agentview_image"])
img = np.flipud(img)
if img.shape[-1] == 4:
    img = img[:, :, :3]
img = img.astype(np.uint8)

cv2.imwrite("outputs/live_agentview_milk.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("saved outputs/live_agentview_milk.jpg")
env.close()
