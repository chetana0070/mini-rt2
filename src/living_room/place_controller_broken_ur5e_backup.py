import os
import numpy as np

from src.living_room.grasp_planner import plan_grasp


STAGE_TO_ID = {
    "reach_above": 0,
    "pregrasp": 1,
    "descend": 2,
    "close_gripper": 3,
    "verify_lift": 4,
    "lift_high": 5,
    "move_above_pallet": 6,
    "lower_vertical": 7,
    "release": 8,
    "retreat_up": 9,
    "abort_failed_grasp": 10,
}


def move_to(obs, target, gripper, action_dim, gain=6.0, max_step=0.65):
    eef = obs["robot0_eef_pos"]
    delta = target - eef

    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = np.clip(delta * gain, -max_step, max_step)

    # Keep orientation fixed for now.
    # Later: put yaw correction in action[3:6] after we test rotation convention.
    if action_dim >= 6:
        action[3:6] = 0.0

    action[-1] = gripper
    return action


def scripted_place_policy(obs, step, env, obj_name, obj_start, bin_pos):
    """
    Perception-style oracle grasp controller.

    Fixes:
    - uses grasp planner
    - follows live XY for large unstable toy/Cereal
    - aborts if object was not lifted
    - never moves low across the table
    """
    # Robot-specific gripper calibration.
    # For Panda, full close can be held.
    # For UR5e/Robotiq, positive close is more like a repeated close command.
    # Holding it during lift/transfer can over-drive contacts and explode physics.
    robot_name = os.environ.get("ROBOT_NAME", "").lower()

    if robot_name == "ur5e":
        # Abort only before lower/release. Do not abort after placement starts.
        abort_window = (360 <= step < 590)
    elif obj_name == "Cereal":
        abort_window = (380 <= step < 460)
    elif obj_name == "Bread":
        abort_window = (430 <= step < 500)
    else:
        abort_window = (280 <= step < 330)

    if abort_window and not lifted_now:
        stage = "abort_failed_grasp"

        if robot_name == "ur5e":
            # Do not suddenly open while touching object.
            # That flicks the mug away.
            abort_grip = hold_gripper
            abort_gain = 1.6
            abort_max_step = 0.06
        else:
            abort_grip = open_gripper
            abort_gain = 4.0
            abort_max_step = 0.50

        action = move_to(
            obs=obs,
            target=plan["above_obj"],
            gripper=abort_grip,
            action_dim=env.action_dim,
            gain=abort_gain,
            max_step=abort_max_step,
        )
        return action, stage

    # Cereal gets slower timing.
    if obj_name == "Cereal":
        if step < 80:
            stage, target, grip, gain = "reach_above", plan["above_obj"], open_gripper, 6.0
        elif step < 170:
            stage, target, grip, gain = "pregrasp", plan["pregrasp"], open_gripper, 4.8
        elif step < 270:
            stage, target, grip, gain = "descend", plan["grasp"], open_gripper, 3.6
        elif step < 330:
            stage, target, grip, gain = "close_gripper", plan["grasp"], close_gripper, 3.2
        elif step < 380:
            stage, target, grip, gain = "verify_lift", plan["lift"], close_gripper, 4.0
        elif step < 460:
            stage, target, grip, gain = "lift_high", plan["lift"], close_gripper, 4.5
        elif step < 570:
            stage, target, grip, gain = "move_above_pallet", plan["above_pallet"], close_gripper, 3.8
        elif step < 650:
            stage, target, grip, gain = "lower_vertical", plan["drop"], close_gripper, 3.8
        elif step < 690:
            stage, target, grip, gain = "release", plan["drop"], open_gripper, 3.2
        else:
            stage, target, grip, gain = "retreat_up", plan["retreat"], open_gripper, 3.8

    else:
        if step < 60:
            stage, target, grip, gain = "reach_above", plan["above_obj"], open_gripper, 7.0
        elif step < 120:
            stage, target, grip, gain = "pregrasp", plan["pregrasp"], open_gripper, 6.5
        elif step < 185:
            stage, target, grip, gain = "descend", plan["grasp"], open_gripper, 5.5
        elif step < 245:
            stage, target, grip, gain = "close_gripper", plan["grasp"], close_gripper, 4.5
        elif step < 280:
            stage, target, grip, gain = "verify_lift", plan["lift"], close_gripper, 5.0
        elif step < 330:
            stage, target, grip, gain = "lift_high", plan["lift"], close_gripper, 5.5
        elif step < 500:
            # Long transfer, especially to robot-left pallet.
            # Move slower to prevent object slipping from gripper.
            stage, target, grip, gain = "move_above_pallet", plan["above_pallet"], close_gripper, 2.8
        elif step < 530:
            safe_drop = plan["drop"].copy()
            if robot_name == "ur5e":
                # UR5e: lower less. Releasing too low caused slip/drop.
                safe_drop[2] += 0.130
                gain = 2.0
            else:
                safe_drop[2] += 0.075
                gain = 2.6
            stage, target, grip = "lower_vertical", safe_drop, close_gripper
        elif step < 575:
            safe_drop = plan["drop"].copy()
            if robot_name == "ur5e":
                safe_drop[2] += 0.130
                gain = 1.8
            else:
                safe_drop[2] += 0.075
                gain = 2.4
            stage, target, grip = "release", safe_drop, open_gripper
        else:
            stage, target, grip, gain = "retreat_up", plan["retreat"], open_gripper, 3.5

    # UR5e generic-object timing override.
    # Goal: longer gentle close, slower lift, later high release.
    if robot_name == "ur5e" and obj_name not in ["Bread", "Cereal"]:
        if step < 70:
            stage, target, grip, gain = "reach_above", plan["above_obj"], open_gripper, 5.5
        elif step < 140:
            stage, target, grip, gain = "pregrasp", plan["pregrasp"], open_gripper, 4.2
        elif step < 200:
            stage, target, grip, gain = "descend", plan["grasp"], open_gripper, 3.0
        elif step < 300:
            # Longer gentle close. Do not move much while closing.
            stage, target, grip, gain = "close_gripper", plan["grasp"], close_gripper, 1.4
        elif step < 360:
            stage, target, grip, gain = "verify_lift", plan["lift"], hold_gripper, 2.4
        elif step < 430:
            stage, target, grip, gain = "lift_high", plan["lift"], hold_gripper, 2.6
        elif step < 590:
            stage, target, grip, gain = "move_above_pallet", plan["above_pallet"], hold_gripper, 2.2
        elif step < 625:
            safe_drop = plan["drop"].copy()
            safe_drop[2] += 0.170
            stage, target, grip, gain = "lower_vertical", safe_drop, hold_gripper, 1.6
        elif step < 670:
            safe_drop = plan["drop"].copy()
            safe_drop[2] += 0.170
            stage, target, grip, gain = "release", safe_drop, open_gripper, 1.4
        else:
            stage, target, grip, gain = "retreat_up", plan["retreat"], open_gripper, 2.5

    # UR5e gripper fix:
    # Close gently during close_gripper.
    # Carry with tiny hold.
    # Open only during release/retreat.
    if robot_name == "ur5e":
        if stage in ["reach_above", "pregrasp", "descend"]:
            grip = open_gripper
        elif stage == "close_gripper":
            grip = close_gripper
        elif stage in [
            "verify_lift",
            "lift_high",
            "move_above_pallet",
            "lower_vertical",
            "abort_failed_grasp",
        ]:
            grip = hold_gripper
        elif stage in ["release", "retreat_up"]:
            grip = open_gripper

    if os.environ.get("ROBOT_NAME", "").lower() == "ur5e":
        stage_max_step = {
            "reach_above": 0.45,
            "pregrasp": 0.30,
            "descend": 0.22,
            "close_gripper": 0.035,
            "verify_lift": 0.08,
            "lift_high": 0.10,
            "move_above_pallet": 0.08,
            "lower_vertical": 0.045,
            "release": 0.035,
            "retreat_up": 0.18,
        }.get(stage, 0.15)
    else:
        stage_max_step = {
            "reach_above": 0.60,
            "pregrasp": 0.45,
            "descend": 0.35,
            "close_gripper": 0.25,
            "verify_lift": 0.35,
            "lift_high": 0.40,
            "move_above_pallet": 0.22,
            "lower_vertical": 0.20,
            "release": 0.16,
            "retreat_up": 0.35,
        }.get(stage, 0.45)

    return move_to(
        obs,
        target,
        grip,
        env.action_dim,
        gain=gain,
        max_step=stage_max_step,
    ), stage


def strict_success(obs, obj_name, obj_start, bin_pos, max_lift_delta, step):
    obj = obs[f"{obj_name}_pos"]

    final_xy = float(np.linalg.norm(obj[:2] - bin_pos[:2]))
    final_z = float(obj[2])
    z0 = float(obj_start[2])

    lifted = max_lift_delta >= 0.10
    placed_xy = final_xy <= 0.065
    settled_height = final_z <= z0 + 0.14

    if obj_name == "Cereal":
        after_release = step >= 695
    else:
        after_release = step >= 630

    ok = lifted and placed_xy and settled_height and after_release

    return ok, {
        "final_xy": final_xy,
        "final_z": final_z,
        "lift_delta": float(max_lift_delta),
        "placed_xy": placed_xy,
        "settled_height": settled_height,
        "after_release": after_release,
    }
