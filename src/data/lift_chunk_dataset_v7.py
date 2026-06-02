import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.action_tokenizer import ActionTokenizer
from src.data.lift_token_dataset import SimpleTextTokenizer
from src.data.lift_token_dataset_v4 import STAGE_TO_ID


class LiftChunkDatasetV7(Dataset):
    def __init__(
        self,
        hdf5_path="datasets/lift_success_demos_50.hdf5",
        camera_key="agentview_image",
        chunk_len=8,
        num_bins=256,
        max_text_len=64,
    ):
        self.hdf5_path = hdf5_path
        self.camera_key = camera_key
        self.chunk_len = chunk_len
        self.action_tokenizer = ActionTokenizer(num_bins=num_bins)
        self.text_tokenizer = SimpleTextTokenizer(max_len=max_text_len)

        self.samples = []

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
                stages = g["stages"][:]

                action_tokens = self.action_tokenizer.encode(actions)
                n = len(actions)

                for t in range(0, n - chunk_len):
                    stage_name = stages[t]
                    if isinstance(stage_name, bytes):
                        stage_name = stage_name.decode("utf-8")

                    stage_id = STAGE_TO_ID[str(stage_name)]

                    step_norm = np.array([t / max(n - 1, 1)], dtype=np.float32)
                    rel = cube_pos[t] - eef_pos[t]

                    proprio = np.concatenate(
                        [eef_pos[t], cube_pos[t], rel, step_norm],
                        axis=0,
                    ).astype(np.float32)

                    chunk_tokens = action_tokens[t : t + chunk_len]
                    chunk_actions = actions[t : t + chunk_len]

                    self.samples.append(
                        {
                            "image": images[t],
                            "text_ids": text_ids,
                            "proprio": proprio,
                            "stage_id": stage_id,
                            "chunk_tokens": chunk_tokens,
                            "chunk_actions": chunk_actions,
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        image = torch.from_numpy(s["image"]).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "text_ids": torch.from_numpy(s["text_ids"]).long(),
            "proprio": torch.from_numpy(s["proprio"]).float(),
            "stage_id": torch.tensor(s["stage_id"], dtype=torch.long),
            "chunk_tokens": torch.from_numpy(s["chunk_tokens"]).long(),
            "chunk_actions": torch.from_numpy(s["chunk_actions"]).float(),
        }
