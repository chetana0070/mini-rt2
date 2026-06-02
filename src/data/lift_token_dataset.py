import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.action_tokenizer import ActionTokenizer


class SimpleTextTokenizer:
    """
    Tiny character tokenizer.
    No internet. No Hugging Face download.
    Good for first Mini RT-2 scaffold.
    """

    def __init__(self, max_len=64):
        self.max_len = max_len

    def encode(self, text):
        text = str(text).lower()
        ids = [min(ord(c), 127) for c in text[: self.max_len]]

        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))

        return np.asarray(ids, dtype=np.int64)


class LiftTokenDataset(Dataset):
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
        self.actions = []
        self.demo_names = []
        self.steps = []

        with h5py.File(hdf5_path, "r") as f:
            for demo_name in sorted(f.keys()):
                g = f[demo_name]

                instruction = g.attrs.get("instruction", f.attrs.get("instruction", "lift the cube"))
                text_ids = self.text_tokenizer.encode(instruction)

                images = g[camera_key][:]
                actions = g["actions"][:]
                action_tokens = self.action_tokenizer.encode(actions)

                for t in range(len(actions)):
                    self.images.append(images[t])
                    self.actions.append(actions[t])
                    self.tokens.append(action_tokens[t])
                    self.text_ids.append(text_ids)
                    self.demo_names.append(demo_name)
                    self.steps.append(t)

        self.images = np.asarray(self.images, dtype=np.uint8)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.tokens = np.asarray(self.tokens, dtype=np.int64)
        self.text_ids = np.asarray(self.text_ids, dtype=np.int64)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        image = self.images[idx]

        # HWC uint8 -> CHW float in [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        text_ids = torch.from_numpy(self.text_ids[idx]).long()
        tokens = torch.from_numpy(self.tokens[idx]).long()
        actions = torch.from_numpy(self.actions[idx]).float()

        return {
            "image": image,
            "text_ids": text_ids,
            "tokens": tokens,
            "actions": actions,
            "idx": torch.tensor(idx, dtype=torch.long),
        }
