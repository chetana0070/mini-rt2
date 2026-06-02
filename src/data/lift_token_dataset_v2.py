import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.action_tokenizer import ActionTokenizer
from src.data.lift_token_dataset import SimpleTextTokenizer


class LiftTokenDatasetV2(Dataset):
    def __init__(
        self,
        hdf5_path="datasets/lift_success_demos.hdf5",
        camera_key="agentview_image",
        num_bins=256,
        max_text_len=64,
    ):
        self.hdf5_path = hdf5_path
        self.camera_key = camera_key
        self.action_tokenizer = ActionTokenizer(num_bins=num_bins)
        self.text_tokenizer = SimpleTextTokenizer(max_len=max_text_len)

        self.images = []
        self.tokens = []
        self.text_ids = []
        self.proprio = []
        self.actions = []

        with h5py.File(hdf5_path, "r") as f:
            for demo_name in sorted(f.keys()):
                g = f[demo_name]

                instruction = g.attrs.get(
                    "instruction",
                    f.attrs.get("instruction", "lift the cube"),
                )

                text_ids = self.text_tokenizer.encode(instruction)

                images = g[camera_key][:]
                actions = g["actions"][:]
                eef_pos = g["eef_pos"][:]
                cube_pos = g["cube_pos"][:]
                action_tokens = self.action_tokenizer.encode(actions)

                n = len(actions)

                for t in range(n):
                    step_norm = np.array([t / max(n - 1, 1)], dtype=np.float32)

                    # proprio = current robot + object + relative vector + time
                    rel = cube_pos[t] - eef_pos[t]
                    prop = np.concatenate(
                        [eef_pos[t], cube_pos[t], rel, step_norm],
                        axis=0,
                    ).astype(np.float32)

                    self.images.append(images[t])
                    self.actions.append(actions[t])
                    self.tokens.append(action_tokens[t])
                    self.text_ids.append(text_ids)
                    self.proprio.append(prop)

        self.images = np.asarray(self.images, dtype=np.uint8)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.tokens = np.asarray(self.tokens, dtype=np.int64)
        self.text_ids = np.asarray(self.text_ids, dtype=np.int64)
        self.proprio = np.asarray(self.proprio, dtype=np.float32)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "text_ids": torch.from_numpy(self.text_ids[idx]).long(),
            "proprio": torch.from_numpy(self.proprio[idx]).float(),
            "tokens": torch.from_numpy(self.tokens[idx]).long(),
            "actions": torch.from_numpy(self.actions[idx]).float(),
        }
