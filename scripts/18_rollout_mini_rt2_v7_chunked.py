import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="checkpoints/mini_rt2_v7_best.pt")
parser.add_argument("--instruction", type=str, default="lift the cube")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--horizon", type=int, default=220)
parser.add_argument("--demo_len", type=int, default=185)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--exec_len", type=int, default=4)
parser.add_argument("--no_render", action="store_true")
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
from src.models.mini_rt2_chunk_policy_v7 import MiniRT2ChunkPolicyV7


def schedule_stage(step):
    if step < 45:
        return 0
    if step < 85:
        return 1
    if step < 125:
        return 2
    if step < 175:
        return 3
    return 4


def preprocess_image(obs):
    img = obs["agentview_image"]
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0


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


def light_filter(action, stage_id, prev_action=None):
    action = action.copy()

    # No rotation for Lift task.
    action[3:6] = 0.0

    # Remove token-128 drift.
    action[np.abs(action) < 0.012] = 0.0

    # Clamp xyz.
    action[0] = np.clip(action[0], -0.45, 0.45)
    action[1] = np.clip(action[1], -0.45, 0.45)
    action[2] = np.clip(action[2], -0.60, 0.60)

    # Use known stage gripper schedule from demos.
    if stage_id in [0, 1, 2]:
        action[-1] = -1.0
    else:
        action[-1] = 1.0

    # Light z safety only.
    if stage_id == 0:
        action[2] = np.clip(action[2], -0.15, 0.25)
    elif stage_id == 1:
        action[2] = np.clip(action[2], -0.35, 0.10)
    elif stage_id == 2:
        action[2] = np.clip(action[2], -0.55, 0.03)
    elif stage_id == 3:
        action[2] = np.clip(action[2], -0.03, 0.03)
    elif stage_id == 4:
        action[2] = max(action[2], 0.20)

    # Small smoothing.
    if prev_action is not None:
        alpha = 0.55
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
    chunk_len = int(ckpt.get("chunk_len", 8))
    proprio_dim = int(ckpt.get("proprio_dim", 10))
    hidden_dim = int(ckpt.get("hidden_dim", 512))

    model = MiniRT2ChunkPolicyV7(
        vocab_size=128,
        num_bins=num_bins,
        action_dim=action_dim,
        num_stages=5,
        chunk_len=chunk_len,
        proprio_dim=proprio_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    action_tokenizer = ActionTokenizer(num_bins=num_bins)
    text_tokenizer = SimpleTextTokenizer(max_len=64)

    text_ids = torch.from_numpy(
        text_tokenizer.encode(args.instruction)
    ).long().unsqueeze(0).to(device)

    render = not args.no_render

    env, camera_names = make_panda_lift_env(
        render=render,
        camera_height=args.height,
        camera_width=args.width,
        horizon=args.horizon,
    )

    camera_keys = [f"{name}_image" for name in camera_names]

    print("Checkpoint:", args.ckpt)
    print("Mode: v7 action chunking")
    print("Chunk len:", chunk_len)
    print("Exec len:", args.exec_len)
    print("Episodes:", args.episodes)
    print("Render:", render)
    print("GL:", args.gl)

    success_count = 0

    for ep in range(args.episodes):
        obs = env.reset()

        start_cube_z = float(obs["cube_pos"][2])
        max_cube_z = start_cube_z

        action_queue = []
        token_queue = []

        prev_action = None

        frames = []
        action_history = []
        token_history = []
        stage_history = []

        success = False

        for step in range(args.horizon):
            stage_id_int = schedule_stage(step)

            if len(action_queue) == 0:
                stage_id = torch.tensor([stage_id_int], dtype=torch.long, device=device)
                image = preprocess_image(obs).to(device)
                proprio = make_proprio(obs, step, args.demo_len).to(device)

                with torch.no_grad():
                    chunk_tokens = model.predict_chunk_tokens(
                        image=image,
                        text_ids=text_ids,
                        proprio=proprio,
                        stage_id=stage_id,
                    )[0].detach().cpu().numpy()

                chunk_actions = action_tokenizer.decode(chunk_tokens)

                usable = min(args.exec_len, len(chunk_actions))

                action_queue = [chunk_actions[i] for i in range(usable)]
                token_queue = [chunk_tokens[i] for i in range(usable)]

            raw_action = action_queue.pop(0)
            pred_tokens = token_queue.pop(0)

            action = light_filter(
                action=raw_action,
                stage_id=stage_id_int,
                prev_action=prev_action,
            )

            prev_action = action.copy()

            obs, reward, done, info = env.step(action)

            cube_z = float(obs["cube_pos"][2])
            max_cube_z = max(max_cube_z, cube_z)
            lift_delta = max_cube_z - start_cube_z

            action_history.append(action)
            token_history.append(pred_tokens)
            stage_history.append(stage_id_int)

            if render:
                env.render()

            stage_name = ID_TO_STAGE[stage_id_int]

            title = (
                f"Mini RT-2 v7 chunk ep={ep} step={step} "
                f"stage={stage_name} dz={lift_delta:.3f}"
            )

            stop = show_camera_panel(
                obs=obs,
                camera_keys=camera_keys,
                window_name="Mini RT-2 v7 Action Chunking",
                title_text=title,
                delay=1,
            )

            frames.append(obs["agentview_image"][::-1])

            if step % 20 == 0:
                print(
                    f"ep={ep} step={step:03d} "
                    f"stage={stage_name:<10} "
                    f"cube_z={cube_z:.3f} dz={lift_delta:.3f} "
                    f"tokens={pred_tokens.tolist()} "
                    f"action={np.round(action, 3).tolist()}"
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

        video_path = f"videos/mini_rt2_v7_chunk_rollout_ep_{ep:04d}.mp4"
        imageio.mimsave(video_path, frames, fps=20)

        action_history = np.asarray(action_history)
        token_history = np.asarray(token_history)

        print()
        print(f"Episode {ep} summary")
        print("success:", success)
        print("max lift delta:", round(float(max_cube_z - start_cube_z), 4))
        print("mean tokens:", np.round(token_history.mean(axis=0), 2).tolist())
        print("mean action:", np.round(action_history.mean(axis=0), 4).tolist())
        print("video:", video_path)
        print()

    env.close()
    close_render_windows()

    print("Final success:", success_count, "/", args.episodes)


if __name__ == "__main__":
    main()
