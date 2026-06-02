import h5py
from collections import Counter
import numpy as np

path = "datasets/lift_success_demos_50.hdf5"

stage_counter = Counter()
demo_lengths = []
final_deltas = []

with h5py.File(path, "r") as f:
    print("Dataset:", path)
    print("Attrs:", dict(f.attrs))
    print("Demos:", len(f.keys()))

    for demo_name in f.keys():
        g = f[demo_name]

        stages = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in g["stages"][:]
        ]

        stage_counter.update(stages)
        demo_lengths.append(g["actions"].shape[0])
        final_deltas.append(float(g.attrs["final_lift_delta"]))

print()
print("Stage counts:")
for stage, count in stage_counter.items():
    print(f"{stage}: {count}")

print()
print("Demo length:")
print("min:", min(demo_lengths))
print("max:", max(demo_lengths))
print("mean:", round(float(np.mean(demo_lengths)), 2))

print()
print("Lift delta:")
print("min:", round(min(final_deltas), 4))
print("max:", round(max(final_deltas), 4))
print("mean:", round(float(np.mean(final_deltas)), 4))

print()
print("Total samples:", sum(stage_counter.values()))
