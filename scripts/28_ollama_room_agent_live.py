import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen2.5:3b")
parser.add_argument("--gl", type=str, default="glfw", choices=["glfw", "egl", "osmesa"])
parser.add_argument("--horizon", type=int, default=600)
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--confirm", action="store_true")
args = parser.parse_args()


OLLAMA_URL = "http://localhost:11434/api/chat"
CONFIG_PATH = "configs/living_room/tasks_v0.json"


OBJECT_ALIASES = {
    "mug": "mug",
    "cup": "mug",
    "milk": "mug",
    "book": "book",
    "bread": "book",
    "remote": "remote",
    "can": "remote",

    # toy/Cereal disabled for now:
    # rectangular object requires reliable orientation-aware grasp controller.
    # "toy": "toy",
    # "box": "toy",
    # "cereal": "toy",
}

TARGET_ALIASES = {
    "robot left pallet": "robot_left_pallet",
    "left pallet": "robot_left_pallet",
    "left tray": "robot_left_pallet",
    "robot left": "robot_left_pallet",
    "left": "robot_left_pallet",
    "bin2": "robot_left_pallet",

    "robot right pallet": "robot_right_pallet",
    "right pallet": "robot_right_pallet",
    "right tray": "robot_right_pallet",
    "robot right": "robot_right_pallet",
    "right": "robot_right_pallet",
    "bin1": "robot_right_pallet",
}


def load_tasks():
    if not Path(CONFIG_PATH).exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    cfg = json.loads(Path(CONFIG_PATH).read_text())

    # Only expose safe enabled tasks to the chatbot.
    # Experimental tasks, like toy/Cereal, stay in config but cannot be executed by agent.
    tasks = [
        t for t in cfg["tasks"]
        if t.get("enabled_for_agent", True) is True
    ]

    task_map = {t["task_id"]: t for t in tasks}
    return cfg, tasks, task_map


def print_tasks(tasks):
    print()
    print("Safe tasks:")
    for t in tasks:
        print(f"  {t['task_id']:<32} | {t['instruction']}")
    print()


def deterministic_parse(user_text, task_map):
    text = user_text.lower().strip()

    if text in {"exit", "quit", "stop", "/bye"}:
        return {
            "action_type": "stop",
            "task_id": "",
            "reply": "Stopping agent.",
            "safety_note": "No robot action executed.",
        }

    if "list" in text and ("task" in text or "skill" in text):
        return {
            "action_type": "list_tasks",
            "task_id": "",
            "reply": "Listing safe tasks.",
            "safety_note": "No robot action executed.",
        }

    obj = ""
    for k, v in OBJECT_ALIASES.items():
        if k in text:
            obj = v
            break

    target = ""
    for k in sorted(TARGET_ALIASES.keys(), key=len, reverse=True):
        if k in text:
            target = TARGET_ALIASES[k]
            break

    if not obj:
        return {
            "action_type": "ask_clarification",
            "task_id": "",
            "reply": "Which object should I move: mug, book, or remote?",
            "safety_note": "No robot action executed.",
        }

    if not target:
        return {
            "action_type": "ask_clarification",
            "task_id": "",
            "reply": "Which pallet should I use: robot-left or robot-right?",
            "safety_note": "No robot action executed.",
        }

    side = "left" if target == "robot_left_pallet" else "right"
    task_id = f"{obj}_to_robot_{side}_pallet"

    if task_id not in task_map:
        return {
            "action_type": "ask_clarification",
            "task_id": "",
            "reply": f"I do not have a safe task for: {obj} to {target}.",
            "safety_note": "No robot action executed.",
        }

    return {
        "action_type": "execute_task",
        "task_id": task_id,
        "reply": f"Executing {task_id}.",
        "safety_note": "Using safe high-lift, move-above-pallet, vertical-lower, release, retreat path.",
    }


def try_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


def ollama_plan(user_text, tasks, task_map):
    allowed = "\n".join([f"- {t['task_id']}: {t['instruction']}" for t in tasks])

    system_prompt = f"""
You are the Mini-RT2-Room robot language planner.

Map the user command to one safe task_id.

Allowed task_ids:
{allowed}

Rules:
- Output JSON only.
- Never invent a task_id.
- robot-left pallet means bin2.
- robot-right pallet means bin1.
- If object is unclear, ask clarification.
- If pallet is unclear, ask clarification.
- Safe objects: mug, book, remote.
JSON format:
{{
  "action_type": "execute_task | ask_clarification | list_tasks | stop",
  "task_id": "...",
  "reply": "...",
  "safety_note": "..."
}}
""".strip()

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096,
        },
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "")
        plan = try_json(content)
    except Exception as e:
        print("Ollama failed. Using deterministic parser.")
        print("Reason:", e)
        return deterministic_parse(user_text, task_map)

    if not isinstance(plan, dict):
        return deterministic_parse(user_text, task_map)

    action_type = plan.get("action_type", "")
    task_id = plan.get("task_id", "")

    if action_type == "execute_task" and task_id in task_map:
        plan.setdefault("reply", f"Executing {task_id}.")
        plan.setdefault("safety_note", "Using safe whitelisted task.")
        return plan

    if action_type in {"ask_clarification", "list_tasks", "stop"}:
        plan.setdefault("task_id", "")
        plan.setdefault("reply", "")
        plan.setdefault("safety_note", "No robot action executed.")
        return plan

    return deterministic_parse(user_text, task_map)


def execute_task(task_id):
    cmd = [
        sys.executable,
        "scripts/26_debug_corrected_object_to_tray_live.py",
        "--task",
        task_id,
        "--gl",
        args.gl,
        "--horizon",
        str(args.horizon),
    ]

    print()
    print("Executing:")
    print(" ".join(cmd))
    print()

    return subprocess.run(cmd)


def main():
    cfg, tasks, task_map = load_tasks()

    print("Mini-RT2-Room Ollama Agent")
    print("Model:", args.model)
    print("Dry run:", args.dry_run)
    print("Live rendering during execution:", not args.dry_run)
    print()
    print("Examples:")
    print("  put the remote in the robot left pallet")
    print("  put the book in the robot right pallet")
    print("  list tasks")
    print("  exit")
    print()

    while True:
        user_text = input("You > ").strip()

        if not user_text:
            continue

        # Deterministic parser first for safety.
        det = deterministic_parse(user_text, task_map)

        # Use Ollama only when deterministic parser cannot execute directly.
        # This keeps robot execution safe.
        if det["action_type"] == "execute_task":
            plan = det
        else:
            plan = ollama_plan(user_text, tasks, task_map)

        print()
        print("Agent plan:")
        print(json.dumps(plan, indent=2))
        print()

        if plan["action_type"] == "stop":
            print("Stopped.")
            break

        if plan["action_type"] == "list_tasks":
            print_tasks(tasks)
            continue

        if plan["action_type"] == "ask_clarification":
            print("Agent:", plan["reply"])
            continue

        if plan["action_type"] == "execute_task":
            task_id = plan["task_id"]

            if args.dry_run:
                print("Dry run. Robot not executed.")
                continue

            if args.confirm:
                ans = input(f"Execute {task_id}? [y/N] ").strip().lower()
                if ans != "y":
                    print("Cancelled.")
                    continue

            result = execute_task(task_id)
            print("Return code:", result.returncode)


if __name__ == "__main__":
    main()
