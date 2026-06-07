import os
import numpy as np


def _side_points(obj_xy, side_name, edge, inset):
    dirs = {
        "front_y_minus": np.array([0.0, -1.0], dtype=np.float32),
        "back_y_plus":   np.array([0.0,  1.0], dtype=np.float32),
        "left_x_minus":  np.array([-1.0, 0.0], dtype=np.float32),
        "right_x_plus":  np.array([1.0,  0.0], dtype=np.float32),
    }

    d = dirs[side_name]
    approach_xy = obj_xy + d * edge
    grasp_xy = obj_xy + d * inset
    return approach_xy.astype(np.float32), grasp_xy.astype(np.float32), side_name


def choose_side_grasp(obs, obj_name, obj_start, bin_pos=None):
    """
    Target-aware side grasp planner.

    Bread/book rule from experiments:
    - robot left pallet / bin2  -> back_y_plus
    - robot right pallet / bin1 -> left_x_minus
    """
    eef = obs["robot0_eef_pos"][:2]
    obj_xy = obj_start[:2].astype(np.float32).copy()

    if obj_name == "Bread":
        edge = 0.055
        inset = 0.010
    elif obj_name == "Can":
        edge = 0.035
        inset = 0.012
    elif obj_name == "Cereal":
        edge = 0.055
        inset = 0.018
    else:
        edge = 0.025
        inset = 0.010

    forced = os.environ.get("FORCE_GRASP_SIDE", "").strip()
    valid = {"front_y_minus", "back_y_plus", "left_x_minus", "right_x_plus"}

    if forced in valid:
        return _side_points(obj_xy, forced, edge, inset)

    if obj_name == "Bread" and bin_pos is not None:
        # bin2 / robot-left pallet has positive y.
        if float(bin_pos[1]) > 0.0:
            return _side_points(obj_xy, "back_y_plus", edge, inset)

        # bin1 / robot-right pallet has negative y.
        return _side_points(obj_xy, "left_x_minus", edge, inset)

    # Fallback scoring for other objects / future experiments.
    candidates = []
    for name in ["front_y_minus", "back_y_plus", "left_x_minus", "right_x_plus"]:
        approach_xy, grasp_xy, _ = _side_points(obj_xy, name, edge, inset)

        dist = float(np.linalg.norm(approach_xy - eef))
        penalty = 0.0

        if approach_xy[1] < -0.43:
            penalty += 5.0
        if approach_xy[0] < -0.20 or approach_xy[0] > 0.34:
            penalty += 5.0

        candidates.append((dist + penalty, name, approach_xy, grasp_xy))

    candidates.sort(key=lambda x: x[0])
    _, name, approach_xy, grasp_xy = candidates[0]
    return approach_xy.astype(np.float32), grasp_xy.astype(np.float32), name
