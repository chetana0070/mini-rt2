import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import robosuite as suite


class ImageBCPolicy(nn.Module):
    def __init__(self, low_dim, action_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.low_mlp = nn.Sequential(
            nn.Linear(low_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, image, low):
        img_feat = self.cnn(image)
        low_feat = self.low_mlp(low)
        x = torch.cat([img_feat, low_feat], dim=-1)
        return self.head(x)


def demo_key(name):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def get_object_pos(env, obs, object_type="milk"):
    for key in ["Milk_pos", "milk_pos", f"{object_type}_pos", "object_pos"]:
        if key in obs:
            arr = np.asarray(obs[key]).reshape(-1)
            if arr.size >= 3 and 0.5 < arr[2] < 1.3:
                return arr[:3].astype(np.float32)

    sim = env.sim

    for name in ["Milk_main", "milk_main", "Milk", "milk"]:
        try:
            bid = sim.model.body_name2id(name)
            pos = sim.data.body_xpos[bid].copy()
            if 0.5 < pos[2] < 1.3:
                return pos.astype(np.float32)
        except Exception:
            pass

    for bid in range(sim.model.nbody):
        name = sim.model.body_id2name(bid)
        if name and object_type.lower() in name.lower():
            pos = sim.data.body_xpos[bid].copy()
            if 0.5 < pos[2] < 1.3:
                return pos.astype(np.float32)

    raise RuntimeError("Could not find object position")


def load_phase_schedule(dataset_path):
    phases = []

    with h5py.File(dataset_path, "r") as f:
        names = sorted(f["data"].keys(), key=demo_key)
        for name in names:
            ph = np.asarray(f["data"][name]["obs"]["phase_id"]).reshape(-1)
            phases.append(ph)

    return np.median(np.stack(phases, axis=0), axis=0).astype(np.float32)


def image_from_obs(obs, camera_name="agentview"):
    key = f"{camera_name}_image"
    if key not in obs:
        raise KeyError(f"Missing image key {key}. Available keys: {list(obs.keys())}")

    img = np.asarray(obs[key])

    if img.ndim != 3:
        raise ValueError(f"Bad image shape: {img.shape}")

    if img.shape[-1] == 3:
        img = np.transpose(img, (2, 0, 1))
    elif img.shape[0] == 3:
        pass
    else:
        raise ValueError(f"Bad image channel shape: {img.shape}")

    img = np.ascontiguousarray(img).astype(np.float32)

    if img.max() > 2.0:
        img = img / 255.0

    return img


def make_low(obs, phase_id, t):
    eef = np.asarray(obs["robot0_eef_pos"]).reshape(-1).astype(np.float32)
    gq = np.asarray(obs["robot0_gripper_qpos"]).reshape(-1).astype(np.float32)
    phase = np.array([phase_id], dtype=np.float32)
    time = np.array([t], dtype=np.float32)

    low = np.concatenate([eef, gq, phase, time], axis=0).astype(np.float32)
    low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
    return low


def resize_images(images, image_size):
    if images.shape[-1] == image_size and images.shape[-2] == image_size:
        return images
    return F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ur5e_bc/image_milk_r140_bc.pt")
    ap.add_argument("--dataset", default="datasets/ur5e_robomimic/processed/ur5e_milk_r140_success_only.hdf5")
    ap.add_argument("--num_trials", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=900)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--allow_rotation", action="store_true")
    ap.add_argument("--binary_gripper", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    model = ImageBCPolicy(ckpt["low_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    low_mean = ckpt["low_mean"].reshape(1, -1)
    low_std = ckpt["low_std"].reshape(1, -1)
    act_mean = ckpt["act_mean"].reshape(1, -1)
    act_std = ckpt["act_std"].reshape(1, -1)
    image_size = int(ckpt.get("image_size", 128))

    phase_schedule = np.rint(load_phase_schedule(args.dataset)).astype(np.float32)
    T_train = len(phase_schedule)

    print("checkpoint:", args.ckpt)
    print("inputs:", ckpt.get("inputs"))
    print("excluded:", ckpt.get("excluded_privileged_inputs"))
    print("device:", device)
    print("image_size:", image_size)

    successes = 0

    for trial in range(args.num_trials):
        env = suite.make(
            env_name="PickPlace",
            robots="UR5e",
            gripper_types="Robotiq140Gripper",
            has_renderer=args.render,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=args.camera,
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
            horizon=args.horizon,
            single_object_mode=2,
            object_type="milk",
            reward_shaping=True,
        )

        obs = env.reset()

        obj0 = get_object_pos(env, obs, "milk")
        target_xy = obj0[:2].copy()

        max_lift = 0.0
        final_xy_dist = 999.0
        success = False

        print(f"\n=== trial {trial} ===")
        print("initial object:", np.round(obj0, 4))

        for step in range(args.horizon):
            obj = get_object_pos(env, obs, "milk")
            lift = float(obj[2] - obj0[2])
            max_lift = max(max_lift, lift)
            final_xy_dist = float(np.linalg.norm(obj[:2] - target_xy))

            t = step / max(1, args.horizon - 1)
            phase_idx = min(T_train - 1, int(t * (T_train - 1)))
            phase_id = float(phase_schedule[phase_idx])

            img = image_from_obs(obs, args.camera)
            low = make_low(obs, phase_id, t)

            low_n = (low.reshape(1, -1) - low_mean) / low_std

            img_t = torch.from_numpy(img).float().unsqueeze(0).to(device)
            low_t = torch.from_numpy(low_n).float().to(device)

            img_t = resize_images(img_t, image_size)

            with torch.no_grad():
                pred_n = model(img_t, low_t).cpu().numpy()

            action = (pred_n * act_std + act_mean).reshape(-1).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)

            if not args.allow_rotation:
                action[3:6] = 0.0

            if args.binary_gripper:
                action[-1] = 1.0 if action[-1] >= 0.0 else -1.0

            obs, reward, done, info = env.step(action)

            if args.render:
                env.render()

            if step % 40 == 0 or step > args.horizon - 80:
                eef = np.asarray(obs["robot0_eef_pos"]).reshape(-1)
                print(
                    f"step={step:03d}",
                    "phase=", round(phase_id, 2),
                    "act=", np.round(action, 3),
                    "obj=", np.round(obj, 3),
                    "eef=", np.round(eef, 3),
                    "lift=", round(lift, 3),
                    "max=", round(max_lift, 3),
                    "xy=", round(final_xy_dist, 3),
                )

            if max_lift > 0.04 and final_xy_dist < 0.08:
                success = True

            if done:
                break

        print(
            "result:",
            "success", success,
            "max_lift", round(max_lift, 4),
            "final_xy_dist", round(final_xy_dist, 4),
        )

        successes += int(success)
        env.close()

    print(f"\nSummary: {successes}/{args.num_trials} image-policy closed-loop successes")


if __name__ == "__main__":
    main()
