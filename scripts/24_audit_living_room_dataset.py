import h5py
import numpy as np

path = "datasets/living_room/living_room_demos_v0_small.hdf5"

with h5py.File(path, "r") as f:
    print("Dataset:", path)
    print("Demos:", len(f.keys()))
    print()

    bad = []

    for k in f.keys():
        g = f[k]

        pos = g["target_object_pos"][:]
        z0 = float(pos[0, 2])
        zmax = float(pos[:, 2].max())
        lift_delta = zmax - z0

        best_xy = float(g.attrs["best_xy_to_bin"])
        task = g.attrs["task_id"]

        ok = lift_delta >= 0.10 and best_xy <= 0.12

        print(
            k,
            task,
            "lift_delta=", round(lift_delta, 4),
            "best_xy=", round(best_xy, 4),
            "OK" if ok else "BAD"
        )

        if not ok:
            bad.append(k)

    print()
    print("Bad demos:", bad)
