from pathlib import Path
import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
import robosuite as suite


class BCMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def get_object_pos(env, obs, object_type="milk"):
    for key in [
        "Milk_pos",
        "milk_pos",
        f"{object_type}_pos",
        "object_pos",
    ]:
        if key in obs:
            arr = np.asarray(obs[key]).reshape(-1)
            if arr.size >= 3 and 0.5 < arr[2] < 1.3:
                return arr[:3].astype(np.float32)

    sim = env.sim
    candidates = [
        "Milk_main",
        "milk_main",
        "Milk",
        "milk",
    ]
    for name in candidates:
        try:
            bid = sim.model.body_name2id(name)
            pos = sim.data.body_xpos[bid].copy()
            if 0.5 < pos[2] < 1.3:
                return pos.astype(np.float32)
        except Exception:
            pass

    # Last fallback: search bodies with milk in name.
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
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for name in names:
            ph = np.asarray(f["data"][name]["obs"]["phase_id"]).reshape(-1)
            phases.append(ph)

    phase_arr = np.stack(phases, axis=0)
    return np.median(phase_arr, axis=0).astype(np.float32)


def make_feature(obs, env, obj_pos, target_xy, phase_id, t):
    eef = np.asarray(obs["robot0_eef_pos"]).reshape(-1).astype(np.float32)
    gq = np.asarray(obs["robot0_gripper_qpos"]).reshape(-1).astype(np.float32)

    parts = [
        eef.reshape(-1),
        gq.reshape(-1),
        obj_pos.reshape(-1),
        target_xy.reshape(-1),
        np.array([phase_id], dtype=np.float32),
        np.array([t], dtype=np.float32),
    ]

    x = np.concatenate(parts, axis=0).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ur5e_bc/lowdim_milk_r140_bc.pt")
    ap.add_argument("--dataset", default="datasets/ur5e_robomimic/processed/ur5e_milk_r140_success_only.hdf5")
    ap.add_argument("--num_trials", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=900)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--allow_rotation", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    model = BCMLP(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_mean = ckpt["x_mean"].reshape(1, -1)
    x_std = ckpt["x_std"].reshape(1, -1)
    y_mean = ckpt["y_mean"].reshape(1, -1)
    y_std = ckpt["y_std"].reshape(1, -1)

    phase_schedule = load_phase_schedule(args.dataset)
    T_train = len(phase_schedule)

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
            camera_names="agentview",
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
        target_xy = obj0[:2].astype(np.float32)

        max_lift = 0.0
        final_xy_dist = 999.0
        success = False

        print(f"\n=== trial {trial} ===")
        print("initial object:", np.round(obj0, 4), "target_xy:", np.round(target_xy, 4))

        for step in range(args.horizon):
            obj = get_object_pos(env, obs, "milk")
            lift = float(obj[2] - obj0[2])
            max_lift = max(max_lift, lift)
            final_xy_dist = float(np.linalg.norm(obj[:2] - target_xy))

            t = step / max(1, args.horizon - 1)
            phase_idx = min(T_train - 1, int(t * (T_train - 1)))
            phase_id = float(phase_schedule[phase_idx])

            x = make_feature(obs, env, obj, target_xy, phase_id, t)
            xn = (x.reshape(1, -1) - x_mean) / x_std

            with torch.no_grad():
                pred_n = model(torch.from_numpy(xn).float().to(device)).cpu().numpy()

            action = (pred_n * y_std + y_mean).reshape(-1).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)

            if not args.allow_rotation:
                action[3:6] = 0.0

            obs, reward, done, info = env.step(action)

            if args.render:
                env.render()

            if step % 40 == 0 or step > args.horizon - 80:
                print(
                    f"step={step:03d}",
                    "phase=", round(phase_id, 2),
                    "act=", np.round(action, 3),
                    "obj=", np.round(obj, 3),
                    "eef=", np.round(np.asarray(obs['robot0_eef_pos']).reshape(-1), 3),
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

    print(f"\nSummary: {successes}/{args.num_trials} closed-loop successes")


if __name__ == "__main__":
    main()
