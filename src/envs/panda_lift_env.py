from robosuite import load_composite_controller_config
import robosuite as suite


def make_panda_lift_env(
    render=True,
    camera_height=128,
    camera_width=128,
    horizon=220,
):
    camera_names = ["agentview", "frontview"]

    controller_config = load_composite_controller_config(
        controller="BASIC",
        robot="Panda",
    )

    # Force right arm to OSC_POSE if key exists.
    try:
        controller_config["body_parts"]["arms"]["right"]["type"] = "OSC_POSE"
    except Exception:
        pass

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=camera_height,
        camera_widths=camera_width,
        control_freq=20,
        horizon=horizon,
        reward_shaping=True,
    )

    return env, camera_names
