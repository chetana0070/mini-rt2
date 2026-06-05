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
    open_gripper = -1.0
    close_gripper = 1.0

    obj_pos = obs[f"{obj_name}_pos"]
    z0 = float(obj_start[2])
    lift_delta_now = float(obj_pos[2] - z0)
    lifted_now = lift_delta_now > 0.08

    plan = plan_grasp(
        obs=obs,
        obj_name=obj_name,
        obj_start=obj_start,
        bin_pos=bin_pos,
        lifted=lifted_now,
    )

    # If grasp failed, do not move empty hand to pallet.
    # Retreat above original object area.
    if step >= 330 and not lifted_now:
        stage = "abort_failed_grasp"
        action = move_to(
            obs=obs,
            target=plan["above_obj"],
            gripper=open_gripper,
            action_dim=env.action_dim,
            gain=4.0,
            max_step=0.50,
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
        elif step < 435:
            stage, target, grip, gain = "move_above_pallet", plan["above_pallet"], close_gripper, 4.2
        elif step < 515:
            stage, target, grip, gain = "lower_vertical", plan["drop"], close_gripper, 4.2
        elif step < 555:
            stage, target, grip, gain = "release", plan["drop"], open_gripper, 3.5
        else:
            stage, target, grip, gain = "retreat_up", plan["retreat"], open_gripper, 4.0

    return move_to(obs, target, grip, env.action_dim, gain=gain), stage


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
        after_release = step >= 560

    ok = lifted and placed_xy and settled_height and after_release

    return ok, {
        "final_xy": final_xy,
        "final_z": final_z,
        "lift_delta": float(max_lift_delta),
        "placed_xy": placed_xy,
        "settled_height": settled_height,
        "after_release": after_release,
    }
