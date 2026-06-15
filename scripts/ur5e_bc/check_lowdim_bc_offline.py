from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn


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


def flatten_obs_array(arr, T):
    arr = np.asarray(arr)
    return arr.reshape(T, -1).astype(np.float32)


def make_features(demo):
    actions = np.asarray(demo["actions"], dtype=np.float32)
    T = actions.shape[0]
    obs = demo["obs"]

    parts = []
    for key in [
        "robot0_eef_pos",
        "robot0_gripper_qpos",
        "object_pos",
        "target_xy",
        "phase_id",
    ]:
        parts.append(flatten_obs_array(obs[key], T))

    t = np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(T, 1)
    parts.append(t)

    X = np.concatenate(parts, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, actions


ckpt_path = Path("runs/ur5e_bc/lowdim_milk_r140_bc.pt")
data_path = Path("datasets/ur5e_robomimic/processed/ur5e_milk_r140_success_only.hdf5")

ckpt = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = BCMLP(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

x_mean = ckpt["x_mean"]
x_std = ckpt["x_std"]
y_mean = ckpt["y_mean"]
y_std = ckpt["y_std"]

with h5py.File(data_path, "r") as f:
    names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))

    for name in names[:5]:
        demo = f["data"][name]
        X, Y = make_features(demo)

        Xn = (X - x_mean) / x_std
        with torch.no_grad():
            pred_n = model(torch.from_numpy(Xn).float().to(device)).cpu().numpy()

        pred = pred_n * y_std + y_mean

        mae = np.mean(np.abs(pred - Y), axis=0)
        maxe = np.max(np.abs(pred - Y), axis=0)

        print("\n", name)
        print("success:", demo.attrs.get("success"), "max_lift:", demo.attrs.get("max_lift"))
        print("MAE per action:", np.round(mae, 5))
        print("MAX err per action:", np.round(maxe, 5))

        for step in [0, 100, 250, 400, 500, 650, 800]:
            print(
                f"step {step:03d}",
                "true", np.round(Y[step], 3),
                "pred", np.round(pred[step], 3),
            )
