import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import h5py
import numpy as np

from src.data.action_tokenizer import ActionTokenizer


def main():
    path = "datasets/lift_success_demos.hdf5"
    tokenizer = ActionTokenizer(num_bins=256, action_min=-1.0, action_max=1.0)

    all_actions = []

    with h5py.File(path, "r") as f:
        print("Dataset:", path)
        print("Attrs:", dict(f.attrs))
        print("Demos:", list(f.keys()))

        for demo_name in f.keys():
            actions = f[demo_name]["actions"][:]
            all_actions.append(actions)

    actions = np.concatenate(all_actions, axis=0)

    tokens = tokenizer.encode(actions)
    recon = tokenizer.decode(tokens)
    err = np.abs(actions - recon)

    print()
    print("Total action samples:", actions.shape[0])
    print("Action shape:", actions.shape)
    print("Token shape:", tokens.shape)

    print()
    print("Action min per dim:", np.round(actions.min(axis=0), 4))
    print("Action max per dim:", np.round(actions.max(axis=0), 4))
    print("Token min per dim:", tokens.min(axis=0))
    print("Token max per dim:", tokens.max(axis=0))

    print()
    print("Mean abs reconstruction error:", round(float(err.mean()), 6))
    print("Max abs reconstruction error:", round(float(err.max()), 6))

    print()
    print("Example action:")
    print(np.round(actions[0], 4))

    print("Example tokens:")
    print(tokens[0])

    print("Decoded action:")
    print(np.round(recon[0], 4))


if __name__ == "__main__":
    main()
