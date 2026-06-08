import os


def ur5e_gripper_command(stage: str, step: int) -> float:
    """
    UR5e / Robotiq scalar gripper command.

    action[-1]:
      negative = open
      positive = close

    Sweep result:
      UR5E_GRASP_Z_ADD=0.020
      UR5E_CLOSE_PEAK=0.030
      UR5E_HOLD=0.004
    """
    open_cmd = float(os.environ.get("UR5E_OPEN", "-0.20"))
    close_peak = float(os.environ.get("UR5E_CLOSE_PEAK", "0.030"))
    hold_cmd = float(os.environ.get("UR5E_HOLD", "0.004"))

    if stage in ["reach_above", "pregrasp", "descend"]:
        return open_cmd

    if stage == "close_gripper":
        if step < 215:
            return 0.006
        elif step < 230:
            return 0.014
        elif step < 245:
            return 0.022
        elif step < 255:
            return close_peak
        else:
            return hold_cmd

    if stage in [
        "verify_lift",
        "lift_high",
        "move_above_pallet",
        "lower_vertical",
        "abort_failed_grasp",
    ]:
        return hold_cmd

    if stage == "release":
        if step < 635:
            return 0.0
        elif step < 655:
            return -0.06
        else:
            return open_cmd

    if stage == "retreat_up":
        return open_cmd

    return hold_cmd
