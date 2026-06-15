from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def lowdim_for_step(demo, step, horizon):
    obs = demo["obs"]

    eef = np.asarray(obs["robot0_eef_pos"][step]).reshape(-1).astype(np.float32)
    gq = np.asarray(obs["robot0_gripper_qpos"][step]).reshape(-1).astype(np.float32)
    phase = np.asarray(obs["phase_id"][step]).reshape(-1).astype(np.float32)
    t = np.array([step / max(1, horizon - 1)], dtype=np.float32)

    return np.concatenate([eef, gq, phase, t], axis=0).astype(np.float32)


def image_for_step(demo, step):
    img = np.asarray(demo["obs"]["agentview_image"][step])

    if img.shape[-1] == 3:
        img = np.transpose(img, (2, 0, 1))
    elif img.shape[0] == 3:
        pass
    else:
        raise ValueError(f"Bad image shape: {img.shape}")

    img = np.ascontiguousarray(img).astype(np.float32)

    if img.max() > 2.0:
        img = img / 255.0

    return img


def resize_images(images, image_size):
    if images.shape[-1] == image_size and images.shape[-2] == image_size:
        return images
    return F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)


ckpt_path = Path("runs/ur5e_bc/image_milk_r140_bc.pt")
data_path = Path("datasets/ur5e_robomimic/processed/ur5e_milk_r140_success_only.hdf5")

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

model = ImageBCPolicy(ckpt["low_dim"], ckpt["action_dim"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

low_mean = ckpt["low_mean"]
low_std = ckpt["low_std"]
act_mean = ckpt["act_mean"]
act_std = ckpt["act_std"]
image_size = int(ckpt.get("image_size", 128))

print("checkpoint:", ckpt_path)
print("inputs:", ckpt.get("inputs"))
print("excluded:", ckpt.get("excluded_privileged_inputs"))
print("image_size:", image_size)

with h5py.File(data_path, "r") as f:
    names = sorted(f["data"].keys(), key=demo_key)

    for name in names[:5]:
        demo = f["data"][name]
        T = demo["actions"].shape[0]
        Y = np.asarray(demo["actions"], dtype=np.float32)

        preds = []

        for step in range(T):
            img = image_for_step(demo, step)
            low = lowdim_for_step(demo, step, T)

            low_n = (low.reshape(1, -1) - low_mean) / low_std

            img_t = torch.from_numpy(img).float().unsqueeze(0).to(device)
            low_t = torch.from_numpy(low_n).float().to(device)

            img_t = resize_images(img_t, image_size)

            with torch.no_grad():
                pred_n = model(img_t, low_t).cpu().numpy()

            pred = pred_n * act_std + act_mean
            preds.append(pred.reshape(-1))

        preds = np.stack(preds, axis=0)

        mae = np.mean(np.abs(preds - Y), axis=0)
        maxe = np.max(np.abs(preds - Y), axis=0)

        print("\n", name)
        print("success:", demo.attrs.get("success"), "max_lift:", demo.attrs.get("max_lift"))
        print("MAE per action:", np.round(mae, 5))
        print("MAX err per action:", np.round(maxe, 5))

        for step in [0, 100, 250, 400, 500, 650, 800]:
            print(
                f"step {step:03d}",
                "true", np.round(Y[step], 3),
                "pred", np.round(preds[step], 3),
            )
