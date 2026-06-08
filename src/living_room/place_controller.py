import os
import numpy as np

from src.living_room.grasp_planner import plan_grasp
from src.living_room.ur5e_gripper_controller import ur5e_gripper_command


STAGE_IDS = {
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
    "abort_object_unstable": 11,
}


def move_to(obs, target, gripper, action_dim, gain=6.0, max_step=0.65):
    eef = obs["robot0_eef_pos"]
    delta = target - eef

    action = np.zeros(action_dim, dtype=np.float32)
    action[:3] = np.clip(delta * gain, -max_step, max_step)

    # Keep orientation fixed for now.
    if action_dim >= 6:
        action[3:6] = 0.0

    action[-1] = gripper
    return action


def scripted_place_policy(obs, step, env, obj_name, obj_start, bin_pos):
    """
    Stage-based pick-place controller.

    Panda:
    - full close can be held.

    UR5e:
    - negative opens
    - positive closes
    - large positive / long close causes contact explosion
    - use gentle close, tiny hold, high release
    """
    robot_name = os.environ.get("ROBOT_NAME", "").lower()

    if robot_name == "ur5e":
        open_gripper = float(os.environ.get("OPEN_GRIPPER", "-0.35"))
        close_gripper = float(os.environ.get("CLOSE_GRIPPER", "0.09"))
        hold_gripper = float(os.environ.get("HOLD_GRIPPER", "0.04"))
    else:
        open_gripper = float(os.environ.get("OPEN_GRIPPER", "-1.0"))
        close_gripper = float(os.environ.get("CLOSE_GRIPPER", "1.0"))
        hold_gripper = close_gripper

    obj_pos = obs[f"{obj_name}_pos"]

    # UR5e safety guard: stop if object already exploded / left workspace.
    if robot_name == "ur5e":
        obj_xy_norm = float(np.linalg.norm(obj_pos[:2]))
        if obj_xy_norm > 1.25 or float(obj_pos[2]) < 0.05:
            action = np.zeros(env.action_dim, dtype=np.float32)
            action[-1] = hold_gripper
            return action, "abort_object_unstable"

    z0 = float(obj_start[2])
    lift_delta_now = float(obj_pos[2] - z0)
    lifted_now = lift_delta_now > 0.08

    # IMPORTANT: plan must exist before any stage uses plan["..."].
    plan = plan_grasp(
        obs=obs,
        obj_name=obj_name,
        obj_start=obj_start,
        bin_pos=bin_pos,
        lifted=lifted_now,
    )

    # Abort only during lift proof window.
    # Do not abort after lower/release, because the object is supposed to come down.
    if robot_name == "ur5e":
        abort_window = (390 <= step < 430)
    elif obj_name == "Cereal":
        abort_window = (380 <= step < 460)
    elif obj_name == "Bread":
        abort_window = (430 <= step < 500)
    else:
        abort_window = (280 <= step < 330)

    if abort_window and not lifted_now:
        stage = "abort_failed_grasp"

        if robot_name == "ur5e":
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

    # Cereal / toy: still experimental.
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
        elif step < 600:
            stage, target, grip, gain = "move_above_pallet", plan["above_pallet"], close_gripper, 3.8
        elif step < 660:
            stage, target, grip, gain = "lower_vertical", plan["drop"], close_gripper, 3.8
        elif step < 695:
            stage, target, grip, gain = "release", plan["drop"], open_gripper, 3.2
        else:
            stage, target, grip, gain = "retreat_up", plan["retreat"], open_gripper, 3.8

    # Bread / book: flat object side grasp.
    elif obj_name == "Bread":
        if step < 70:
            stage, target, grip, gain = "reach_above", plan["above_obj"], open_gripper, 6.0
        elif step < 160:
            stage, target, grip, gain = "pregrasp", plan["pregrasp"], open_gripper, 4.5
        elif step < 300:
            stage, target, grip, gain = "descend", plan["grasp"], open_gripper, 3.2
        elif step < 380:
            stage, target, grip, gain = "close_gripper", plan["grasp"], close_gripper, 2.8
        elif step < 430:
            stage, target, grip, gain = "verify_lift", plan["lift"], close_gripper, 3.5
        elif step < 500:
            stage, target, grip, gain = "lift_high", plan["lift"], close_gripper, 4.0
        elif step < 650:
            stage, target, grip, gain = "move_above_pallet", plan["above_pallet"], close_gripper, 2.6
        elif step < 735:
            stage, target, grip, gain = "lower_vertical", plan["drop"], close_gripper, 3.0
        elif step < 775:
            stage, target, grip, gain = "release", plan["drop"], open_gripper, 2.8
        else:
            stage, target, grip, gain = "retreat_up", plan["retreat"], open_gripper, 3.4

    # Generic objects: Milk/mug and Can/remote.
    else:
        if robot_name == "ur5e":
            if step < 70:
                stage, target, grip, gain = "reach_above", plan["above_obj"], open_gripper, 5.5
            elif step < 140:
                stage, target, grip, gain = "pregrasp", plan["pregrasp"], open_gripper, 4.2
            elif step < 200:
                stage, target, grip, gain = "descend", plan["grasp"], open_gripper, 3.0
            elif step < 260:
                # UR5e: freeze arm while closing fingers.
                # Close briefly only. Long close causes object explosion.
                hold_pose = obs["robot0_eef_pos"].copy()
                stage, target, grip, gain = "close_gripper", hold_pose, close_gripper, 0.0
            elif step < 350:
                # Begin lift slowly with tiny hold, not close command.
                stage, target, grip, gain = "verify_lift", plan["lift"], hold_gripper, 1.8
            elif step < 430:
                stage, target, grip, gain = "lift_high", plan["lift"], hold_gripper, 2.6
            elif step < 610:
                # UR5e place bias:
                # Mug lands with +x drift after release, so release slightly left.
                place_target = plan["above_pallet"].copy()
                place_target[0] -= float(os.environ.get("UR5E_PLACE_BIAS_X", "0.08"))
                place_target[1] += float(os.environ.get("UR5E_PLACE_BIAS_Y", "0.00"))
                stage, target, grip, gain = "move_above_pallet", place_target, hold_gripper, 2.2
            elif step < 670:
                place_target = plan["above_pallet"].copy()
                place_target[0] -= float(os.environ.get("UR5E_PLACE_BIAS_X", "0.08"))
                place_target[1] += float(os.environ.get("UR5E_PLACE_BIAS_Y", "0.00"))
                stage, target, grip, gain = "release", place_target, open_gripper, 1.2
            else:
                stage, target, grip, gain = "retreat_up", plan["retreat"], open_gripper, 2.5

        else:
            if step < 60:
                stage, target, grip, gain = "reach_above", plan["above_obj"], open_gripper, 7.0
            elif step < 120:
                stage, target, grip, gain = "pregrasp", plan["pregrasp"], open_gripper, 6.5
            elif step < 190:
                stage, target, grip, gain = "descend", plan["grasp"], open_gripper, 5.5
            elif step < 250:
                stage, target, grip, gain = "close_gripper", plan["grasp"], close_gripper, 4.5
            elif step < 280:
                stage, target, grip, gain = "verify_lift", plan["lift"], close_gripper, 5.0
            elif step < 330:
                stage, target, grip, gain = "lift_high", plan["lift"], close_gripper, 5.5
            elif step < 500:
                stage, target, grip, gain = "move_above_pallet", plan["above_pallet"], close_gripper, 2.8
            elif step < 585:
                stage, target, grip, gain = "lower_vertical", plan["drop"], close_gripper, 3.2
            elif step < 625:
                stage, target, grip, gain = "release", plan["drop"], open_gripper, 2.8
            else:
                stage, target, grip, gain = "retreat_up", plan["retreat"], open_gripper, 3.5

    # UR5e gripper rule:
    # Use a finger trajectory, not one constant gripper command.
    if robot_name == "ur5e":
        grip = ur5e_gripper_command(stage, step)

    if robot_name == "ur5e":
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
        }.get(stage, 0.10)
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
        obs=obs,
        target=target,
        gripper=grip,
        action_dim=env.action_dim,
        gain=gain,
        max_step=stage_max_step,
    ), stage


def strict_success(obs, obj_name, obj_start, bin_pos, max_lift_delta, step):
    obj_pos = obs[f"{obj_name}_pos"]

    final_xy = float(np.linalg.norm(obj_pos[:2] - bin_pos[:2]))
    final_z = float(obj_pos[2])
    lift_delta = float(max_lift_delta)

    placed_xy = final_xy < 0.065
    settled_height = final_z < float(obj_start[2]) + 0.20

    robot_name = os.environ.get("ROBOT_NAME", "").lower()

    if obj_name == "Cereal":
        after_release = step >= 695
    elif obj_name == "Bread":
        after_release = step >= 780
    elif robot_name == "ur5e":
        after_release = step >= 675
    else:
        after_release = step >= 630

    success = bool(
        placed_xy
        and settled_height
        and after_release
        and lift_delta > 0.08
    )

    metrics = {
        "final_xy": final_xy,
        "final_z": final_z,
        "lift_delta": lift_delta,
        "placed_xy": placed_xy,
        "settled_height": settled_height,
        "after_release": after_release,
    }

    return success, metrics
