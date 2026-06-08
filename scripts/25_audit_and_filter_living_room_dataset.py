import argparse
import h5py
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", type=str, default="datasets/living_room/living_room_demos_v0_60.hdf5")
parser.add_argument("--out_path", type=str, default="datasets/living_room/living_room_demos_v0_60_clean.hdf5")
parser.add_argument("--min_lift_delta", type=float, default=0.10)
parser.add_argument("--max_best_xy", type=float, default=0.12)
args = parser.parse_args()


def copy_group(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value

    for name, item in src.items():
        if isinstance(item, h5py.Dataset):
            src.copy(item, dst, name=name)
        elif isinstance(item, h5py.Group):
            sub = dst.create_group(name)
            copy_group(item, sub)


def main():
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    bad = []
    task_counts = {}

    with h5py.File(in_path, "r") as f:
        print("Input:", in_path)
        print("Demos:", len(f.keys()))
        print()

        for k in sorted(f.keys()):
            g = f[k]

            pos = g["target_object_pos"][:]
            z0 = float(pos[0, 2])
            zmax = float(pos[:, 2].max())
            lift_delta = zmax - z0

            best_xy = float(g.attrs["best_xy_to_bin"])
            task = str(g.attrs["task_id"])

            ok = lift_delta >= args.min_lift_delta and best_xy <= args.max_best_xy

            print(
                k,
                task,
                "lift_delta=", round(lift_delta, 4),
                "best_xy=", round(best_xy, 4),
                "OK" if ok else "BAD",
            )

            if ok:
                kept.append(k)
                task_counts[task] = task_counts.get(task, 0) + 1
            else:
                bad.append(k)

        print()
        print("Kept:", len(kept))
        print("Bad:", len(bad))
        print("Task counts:", task_counts)
        print("Bad demos:", bad)

        with h5py.File(out_path, "w") as out:
            for key, value in f.attrs.items():
                out.attrs[key] = value

            out.attrs["source_dataset"] = str(in_path)
            out.attrs["filter_min_lift_delta"] = args.min_lift_delta
            out.attrs["filter_max_best_xy"] = args.max_best_xy
            out.attrs["clean_demos"] = len(kept)
            out.attrs["bad_demos"] = len(bad)

            for new_i, old_k in enumerate(kept):
                new_k = f"demo_{new_i:05d}"
                src = f[old_k]
                dst = out.create_group(new_k)
                copy_group(src, dst)
                dst.attrs["source_demo"] = old_k

    print()
    print("Wrote clean dataset:", out_path)


if __name__ == "__main__":
    main()
