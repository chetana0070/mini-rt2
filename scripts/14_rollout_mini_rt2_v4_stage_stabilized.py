import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="checkpoints/mini_rt2_v4_best.pt")
parser.add_argument("--instruction", type=str, default="lift the cube")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--horizon", type=int, default=220)
parser.add_argument("--demo_len", type=int, default=185)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--no_render", action="store_true")
parser.add_argument("--action_scale", type=float, default=1.0)
args = parser.parse_args()

os.environ["MUJOCO_GL"] = args.gl

import imageio
import numpy as np
import torch

from src.data.action_tokenizer import ActionTokenizer
from src.data.lift_token_dataset import SimpleTextTokenizer
from src.data.lift_token_dataset_v4 import ID_TO_STAGE
from src.envs.panda_lift_env import make_panda_lift_env
from src.envs.render_utils import show_camera_panel, close_render_windows
from src.models.mini_rt2_policy_v4 import MiniRT2StagePolicyV4


def schedule_stage(step):
    """
    Same stage timing as the successful scripted demos:
    above:      0-44
    pregrasp:  45-84
    descend:   85-124
    close_hold:125-174
    slow_lift: 175+
    """
    if step < 45:
        return 0
    if step < 85:
        return 1
    if step < 125:
        return 2
    if step < 175:
        return 3
    return 4


def preprocess_image(obs, camera_key):
    img = obs[camera_key]
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def make_proprio(obs, step, demo_len):
    eef = obs["robot0_eef_pos"].astype(np.float32)
    cube = obs["cube_pos"].astype(np.float32)
    rel = cube - eef

    step_norm = np.array(
        [min(step / max(demo_len - 1, 1), 1.0)],
        dtype=np.float32,
    )

    prop = np.concatenate([eef, cube, rel, step_norm], axis=0)
    return torch.from_numpy(prop).float().unsqueeze(0)


def stabilize_by_stage(raw_action, stage_id, pred_tokens, prev_action):
    action = raw_action.copy()

    # rotation unused
    action[3:6] = 0.0

    # remove tiny token drift
    action[np.abs(action) < 0.012] = 0.0

    # clamp xyz
    action[0] = np.clip(action[0], -0.45, 0.45)
    action[1] = np.clip(action[1], -0.45, 0.45)
    action[2] = np.clip(action[2], -0.60, 0.60)

    # stage-aware gripper
    if stage_id in [0, 1, 2]:
        action[-1] = -1.0
    else:
        action[-1] = 1.0

    # stage-aware z guard
    if stage_id == 0:
        # approach above cube, do not dive too fast
        action[2] = np.clip(action[2], -0.15, 0.25)

    elif stage_id == 1:
        # pregrasp, allow controlled descend
        action[2] = np.clip(action[2], -0.35, 0.10)

    elif stage_id == 2:
        # descend, mostly downward or stable
        action[2] = min(action[2], 0.03)
        action[2] = max(action[2], -0.60)

    elif stage_id == 3:
        # close hold, do not lift too early
        action[2] = np.clip(action[2], -0.03, 0.03)

    elif stage_id == 4:
        # lift
        action[2] = max(action[2], 0.20)

    # smooth xyz
    if prev_action is not None:
        alpha = 0.40
        action[:3] = alpha * action[:3] + (1.0 - alpha) * prev_action[:3]

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def main():
    Path("videos").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    ckpt = torch.load(args.ckpt, map_location=device)

    num_bins = int(ckpt.get("num_bins", 256))
    action_dim = int(ckpt.get("action_dim", 7))
    proprio_dim = int(ckpt.get("proprio_dim", 10))
    num_stages = int(ckpt.get("num_stages", 5))
    hidden_dim = int(ckpt.get("hidden_dim", 384))

    model = MiniRT2StagePolicyV4(
        vocab_size=128,
        num_bins=num_bins,
        action_dim=action_dim,
        num_stages=num_stages,
        proprio_dim=proprio_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    action_tokenizer = ActionTokenizer(num_bins=num_bins)
    text_tokenizer = SimpleTextTokenizer(max_len=64)

    text_ids_np = text_tokenizer.encode(args.instruction)
    text_ids = torch.from_numpy(text_ids_np).long().unsqueeze(0).to(device)

    render = not args.no_render

    env, camera_names = make_panda_lift_env(
        render=render,
        camera_height=args.height,
        camera_width=args.width,
        horizon=args.horizon,
    )

    camera_keys = [f"{name}_image" for name in camera_names]

    print("Checkpoint:", args.ckpt)
    print("Instruction:", args.instruction)
    print("Episodes:", args.episodes)
    print("Mode: v4 stage-conditioned stabilized")
    print("Render:", render)
    print("GL:", args.gl)

    success_count = 0

    for ep in range(args.episodes):
        obs = env.reset()

        start_cube_z = float(obs["cube_pos"][2])
        max_cube_z = start_cube_z

        frames = []
        token_history = []
        action_history = []
        stage_history = []
        pred_stage_history = []

        prev_action = None
        success = False

        for step in range(args.horizon):
            stage_id_int = schedule_stage(step)
            stage_id = torch.tensor([stage_id_int], dtype=torch.long, device=device)

            image = preprocess_image(obs, "agentview_image").to(device)
            proprio = make_proprio(obs, step, args.demo_len).to(device)

            with torch.no_grad():
                pred_tokens, pred_stage = model.predict(
                    image=image,
                    text_ids=text_ids,
                    proprio=proprio,
                    stage_id=stage_id,
                )

            pred_tokens = pred_tokens[0].detach().cpu().numpy()
            pred_stage_id = int(pred_stage[0].detach().cpu().item())

            raw_action = action_tokenizer.decode(pred_tokens)
            raw_action = np.clip(raw_action * args.action_scale, -1.0, 1.0).astype(np.float32)

            action = stabilize_by_stage(
                raw_action=raw_action,
                stage_id=stage_id_int,
                pred_tokens=pred_tokens,
                prev_action=prev_action,
            )

            prev_action = action.copy()

            obs, reward, done, info = env.step(action)

            cube_z = float(obs["cube_pos"][2])
            max_cube_z = max(max_cube_z, cube_z)
            lift_delta = max_cube_z - start_cube_z

            token_history.append(pred_tokens)
            action_history.append(action)
            stage_history.append(stage_id_int)
            pred_stage_history.append(pred_stage_id)

            if render:
                env.render()

            stage_name = ID_TO_STAGE[stage_id_int]
            pred_stage_name = ID_TO_STAGE.get(pred_stage_id, "unknown")

            title = (
                f"Mini RT-2 v4 ep={ep} step={step} stage={stage_name} "
                f"pred_stage={pred_stage_name} dz={lift_delta:.3f}"
            )

            stop = show_camera_panel(
                obs=obs,
                camera_keys=camera_keys,
                window_name="Mini RT-2 v4 Stage Rollout",
                title_text=title,
                delay=1,
            )

            frames.append(obs["agentview_image"][::-1])

            if step % 20 == 0:
                print(
                    f"ep={ep} step={step:03d} "
                    f"stage={stage_name:<10} pred_stage={pred_stage_name:<10} "
                    f"cube_z={cube_z:.3f} dz={lift_delta:.3f} "
                    f"raw={np.round(raw_action, 3).tolist()} "
                    f"stable={np.round(action, 3).tolist()} "
                    f"tokens={pred_tokens.tolist()}"
                )

            if lift_delta >= 0.08:
                success = True
                success_count += 1
                print(f"Episode {ep}: SUCCESS at step {step}, lift_delta={lift_delta:.4f}")
                break

            if done or stop:
                print(f"Episode {ep}: stopped at step {step}, lift_delta={lift_delta:.4f}")
                break

            time.sleep(0.01)

        video_path = f"videos/mini_rt2_v4_stage_rollout_ep_{ep:04d}.mp4"
        imageio.mimsave(video_path, frames, fps=20)

        token_history = np.asarray(token_history)
        action_history = np.asarray(action_history)
        stage_history = np.asarray(stage_history)
        pred_stage_history = np.asarray(pred_stage_history)

        stage_match = float((stage_history == pred_stage_history).mean()) if len(stage_history) > 0 else 0.0

        print()
        print(f"Episode {ep} summary")
        print("success:", success)
        print("max lift delta:", round(float(max_cube_z - start_cube_z), 4))
        print("stage_match:", round(stage_match, 3))
        print("mean tokens:", np.round(token_history.mean(axis=0), 2).tolist())
        print("mean action:", np.round(action_history.mean(axis=0), 4).tolist())
        print("video:", video_path)
        print()

    env.close()
    close_render_windows()

    print("Final success:", success_count, "/", args.episodes)


if __name__ == "__main__":
    main()
