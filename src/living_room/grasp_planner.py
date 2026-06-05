import numpy as np


OBJECT_META = {
    "Milk": {
        "alias": "mug",
        "grasp_z_offset": 0.010,
        "pregrasp_z_offset": 0.120,
        "place_z_offset": 0.055,
        "safe_z_add": 0.360,
        "use_live_xy_until_lift": False,
        "xy_bias": [0.0, 0.0],
        "yaw_cmd": 0.0,
    },
    "Bread": {
        "alias": "book",
        "grasp_z_offset": 0.005,
        "pregrasp_z_offset": 0.110,
        "place_z_offset": 0.035,
        "safe_z_add": 0.360,
        "use_live_xy_until_lift": False,
        "xy_bias": [0.0, 0.0],
        "yaw_cmd": 0.0,
    },
    "Can": {
        "alias": "remote",
        "grasp_z_offset": 0.005,
        "pregrasp_z_offset": 0.110,
        "place_z_offset": 0.035,
        "safe_z_add": 0.360,
        "use_live_xy_until_lift": False,
        "xy_bias": [0.0, 0.0],
        "yaw_cmd": 0.0,
    },
    "Cereal": {
        "alias": "toy",
        "grasp_z_offset": 0.035,
        "pregrasp_z_offset": 0.150,
        "place_z_offset": 0.080,
        "safe_z_add": 0.400,
        "use_live_xy_until_lift": True,
        "xy_bias": [0.0, 0.0],
        "yaw_cmd": -0.25
    },
}


def plan_grasp(obs, obj_name, obj_start, bin_pos, lifted=False):
    meta = OBJECT_META.get(obj_name, OBJECT_META["Can"])

    current_obj = obs[f"{obj_name}_pos"].copy()

    if meta["use_live_xy_until_lift"] and not lifted:
        base_xy = current_obj[:2].copy()
    else:
        base_xy = obj_start[:2].copy()

    base_xy = base_xy + np.asarray(meta["xy_bias"], dtype=np.float32)

    z0 = float(obj_start[2])
    safe_z = max(z0 + float(meta["safe_z_add"]), 1.20)

    above_obj = np.array([base_xy[0], base_xy[1], safe_z])
    pregrasp = np.array([base_xy[0], base_xy[1], z0 + float(meta["pregrasp_z_offset"])])
    grasp = np.array([base_xy[0], base_xy[1], z0 + float(meta["grasp_z_offset"])])
    lift = np.array([base_xy[0], base_xy[1], safe_z])
    above_pallet = np.array([bin_pos[0], bin_pos[1], safe_z])
    drop = np.array([bin_pos[0], bin_pos[1], z0 + float(meta["place_z_offset"])])
    retreat = np.array([bin_pos[0], bin_pos[1], safe_z])

    return {
        "object": obj_name,
        "alias": meta["alias"],
        "current_obj": current_obj,
        "above_obj": above_obj,
        "pregrasp": pregrasp,
        "grasp": grasp,
        "lift": lift,
        "above_pallet": above_pallet,
        "drop": drop,
        "retreat": retreat,
        "yaw_cmd": float(meta["yaw_cmd"]),
        "safe_z": safe_z,
    }
