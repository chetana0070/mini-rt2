import robosuite as suite
from robosuite import load_composite_controller_config


OBJECT_ALIASES = {
    "Milk": "mug",
    "Bread": "book",
    "Cereal": "box",
    "Can": "remote",
}


def make_pickplace_env(
    render=True,
    camera_height=128,
    camera_width=128,
    horizon=260,
    robot="UR5e",
):
    controller_config = load_composite_controller_config(
        controller="BASIC",
        robot=robot,
    )

    env = suite.make(
        env_name="PickPlace",
        robots=robot,
        controller_configs=controller_config,
        has_renderer=render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "frontview"],
        camera_heights=camera_height,
        camera_widths=camera_width,
        reward_shaping=True,
        control_freq=20,
        horizon=horizon,
    )

    return env, ["agentview", "frontview"]


def get_body_pos(env, body_name):
    body_id = env.sim.model.body_name2id(body_name)
    return env.sim.data.body_xpos[body_id].copy()


def get_object_pos(obs, obj_name):
    return obs[f"{obj_name}_pos"].copy()


def get_bin_pos(env, bin_name="bin1"):
    return get_body_pos(env, bin_name)
