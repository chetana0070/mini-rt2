import os
import numpy as np
from src.living_room.side_grasp_planner import choose_side_grasp


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
        # Flat object: go lower and approach more carefully.
        "grasp_z_offset": -0.020,
        "pregrasp_z_offset": 0.090,
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

    # UR5e calibration:
    # robot0_eef_pos is not perfectly centered between Robotiq fingers.
    # In logs, EEF closed about +x, -y from mug center.
    # So command EEF slightly -x, +y to align fingers on object.
    if os.environ.get("ROBOT_NAME", "").lower() == "ur5e" and obj_name in ["Milk", "Can"]:
        ur5e_bias = np.array([
            float(os.environ.get("UR5E_GRASP_BIAS_X", "-0.035")),
            float(os.environ.get("UR5E_GRASP_BIAS_Y", "0.025")),
        ], dtype=np.float32)
        base_xy = base_xy + ur5e_bias

    z0 = float(obj_start[2])
    # Robot-specific carrying height.
    # Panda tolerated high carry around 1.20+.
    # UR5e becomes unstable when carrying too high with the Robotiq gripper.
    if os.environ.get("ROBOT_NAME", "").lower() == "ur5e":
        safe_z = max(z0 + 0.240, 1.08)
    else:
        safe_z = max(z0 + float(meta["safe_z_add"]), 1.20)

    side_label = "center_top"
    approach_xy = base_xy.copy()
    grasp_xy = base_xy.copy()

    if obj_name == "Bread":
        # Flat object: choose reachable side, approach edge, then move inward.
        approach_xy, grasp_xy, side_label = choose_side_grasp(obs, obj_name, obj_start, bin_pos=bin_pos)

    above_obj = np.array([approach_xy[0], approach_xy[1], safe_z])
    if os.environ.get("ROBOT_NAME", "").lower() == "ur5e" and obj_name in ["Milk", "Can"]:
        # UR5e Robotiq is larger than Panda gripper.
        # Keep palm/fingers higher to avoid contact explosion.
        pregrasp_z = z0 + float(os.environ.get("UR5E_PREGRASP_Z_ADD", "0.145"))
        grasp_z = z0 + float(os.environ.get("UR5E_GRASP_Z_ADD", "0.020"))
    else:
        pregrasp_z = z0 + float(meta["pregrasp_z_offset"])
        grasp_z = z0 + float(meta["grasp_z_offset"])

    pregrasp = np.array([approach_xy[0], approach_xy[1], pregrasp_z])
    grasp = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
    lift = np.array([grasp_xy[0], grasp_xy[1], safe_z])
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
        "side_label": side_label,
        "approach_xy": approach_xy,
        "grasp_xy": grasp_xy,
        "safe_z": safe_z,
    }
