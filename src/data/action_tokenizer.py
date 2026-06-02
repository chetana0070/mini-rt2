import numpy as np


class ActionTokenizer:
    """
    RT-2-style action tokenizer.

    Continuous action range:
        [-1, 1]

    Token range:
        [0, num_bins - 1]
    """

    def __init__(self, num_bins=256, action_min=-1.0, action_max=1.0):
        self.num_bins = int(num_bins)
        self.action_min = float(action_min)
        self.action_max = float(action_max)

    def encode(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, self.action_min, self.action_max)

        scaled = (actions - self.action_min) / (self.action_max - self.action_min)
        tokens = np.round(scaled * (self.num_bins - 1)).astype(np.int64)

        return tokens

    def decode(self, tokens):
        tokens = np.asarray(tokens, dtype=np.int64)
        tokens = np.clip(tokens, 0, self.num_bins - 1)

        scaled = tokens.astype(np.float32) / float(self.num_bins - 1)
        actions = scaled * (self.action_max - self.action_min) + self.action_min

        return actions.astype(np.float32)

    def reconstruction_error(self, actions):
        tokens = self.encode(actions)
        recon = self.decode(tokens)
        err = np.abs(actions - recon)

        return {
            "mean_abs_error": float(err.mean()),
            "max_abs_error": float(err.max()),
        }
