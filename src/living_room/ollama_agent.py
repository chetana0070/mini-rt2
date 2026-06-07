import json
import re
from pathlib import Path

import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"


COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [
                "execute_task",
                "ask_clarification",
                "list_tasks",
                "status",
                "stop"
            ],
        },
        "task_id": {"type": "string"},
        "object": {"type": "string"},
        "target": {"type": "string"},
        "reply": {"type": "string"},
        "safety_note": {"type": "string"},
    },
    "required": [
        "action_type",
        "task_id",
        "object",
        "target",
        "reply",
        "safety_note"
    ],
}


OBJECT_ALIASES = {
    "mug": "mug",
    "cup": "mug",
    "milk": "mug",

    "book": "book",
    "bread": "book",

    "remote": "remote",
    "can": "remote",
}


TARGET_ALIASES = {
    "left": "robot_left_pallet",
    "robot left": "robot_left_pallet",
    "left pallet": "robot_left_pallet",
    "robot left pallet": "robot_left_pallet",
    "bin2": "robot_left_pallet",

    "right": "robot_right_pallet",
    "robot right": "robot_right_pallet",
    "right pallet": "robot_right_pallet",
    "robot right pallet": "robot_right_pallet",
    "bin1": "robot_right_pallet",
}


def load_tasks(config_path="configs/living_room/tasks_v0.json"):
    cfg = json.loads(Path(config_path).read_text())
    tasks = cfg["tasks"]
    task_map = {t["task_id"]: t for t in tasks}
    return cfg, tasks, task_map


def task_summary(tasks):
    lines = []
    for t in tasks:
        lines.append(
            f"- {t['task_id']}: {t['instruction']} "
            f"(object={t['living_room_object']}, target={t['target_container']})"
        )
    return "\n".join(lines)


def build_system_prompt(tasks):
    return f"""
You are the language planner for Mini-RT2-Room.

Your job:
Map the user's natural language command to exactly one safe robot task.

Allowed task IDs:
{task_summary(tasks)}

Rules:
1. Never invent a task_id.
2. Only choose from allowed task IDs.
3. Robot-left pallet means bin2.
4. Robot-right pallet means bin1.
5. If the user does not specify left or right pallet, ask for clarification.
6. If the object is not mug, book, or remote, ask for clarification.
7. If the user asks for unsafe behavior, refuse and ask for a safe pallet task.
8. Output JSON only using the given schema.

Examples:
User: put the remote on the left pallet
Output task_id: remote_to_robot_left_pallet

User: move the book to the robot right pallet
Output task_id: book_to_robot_right_pallet

User: place the cup in the left tray
Output task_id: mug_to_robot_left_pallet
""".strip()


def try_parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def deterministic_parse(user_text, task_map):
    text = user_text.lower().strip()

    if text in {"exit", "quit", "stop"}:
        return {
            "action_type": "stop",
            "task_id": "",
            "object": "",
            "target": "",
            "reply": "Stopping the agent.",
            "safety_note": "No robot action executed.",
        }

    if "list" in text and ("task" in text or "skill" in text):
        return {
            "action_type": "list_tasks",
            "task_id": "",
            "object": "",
            "target": "",
            "reply": "Here are the available safe tasks.",
            "safety_note": "Listing tasks only.",
        }

    obj = ""
    for key, value in OBJECT_ALIASES.items():
        if key in text:
            obj = value
            break

    target = ""
    # Prefer longer phrases first.
    for key in sorted(TARGET_ALIASES.keys(), key=len, reverse=True):
        if key in text:
            target = TARGET_ALIASES[key]
            break

    if not obj:
        return {
            "action_type": "ask_clarification",
            "task_id": "",
            "object": "",
            "target": target,
            "reply": "Which object should I move: mug, book, or remote?",
            "safety_note": "No robot action executed.",
        }

    if not target:
        return {
            "action_type": "ask_clarification",
            "task_id": "",
            "object": obj,
            "target": "",
            "reply": "Which pallet should I use: robot-left or robot-right?",
            "safety_note": "No robot action executed.",
        }

    side = "left" if target == "robot_left_pallet" else "right"
    task_id = f"{obj}_to_robot_{side}_pallet"

    if task_id not in task_map:
        return {
            "action_type": "ask_clarification",
            "task_id": "",
            "object": obj,
            "target": target,
            "reply": f"I do not have a safe skill for {obj} to {target}.",
            "safety_note": "No robot action executed.",
        }

    return {
        "action_type": "execute_task",
        "task_id": task_id,
        "object": obj,
        "target": target,
        "reply": f"Executing safe skill: {task_id}.",
        "safety_note": "Using collision-safe high-lift, move-above-pallet, vertical-lower, release, retreat path.",
    }


def validate_plan(plan, task_map, user_text):
    if not isinstance(plan, dict):
        return deterministic_parse(user_text, task_map)

    action_type = plan.get("action_type", "")

    if action_type == "execute_task":
        task_id = plan.get("task_id", "")
        if task_id in task_map:
            plan.setdefault("reply", f"Executing {task_id}.")
            plan.setdefault("safety_note", "Using safe whitelisted task.")
            return plan

        # LLM gave invalid task. Use deterministic safety fallback.
        return deterministic_parse(user_text, task_map)

    if action_type in {"ask_clarification", "list_tasks", "status", "stop"}:
        plan.setdefault("task_id", "")
        plan.setdefault("object", "")
        plan.setdefault("target", "")
        plan.setdefault("reply", "")
        plan.setdefault("safety_note", "No robot action executed.")
        return plan

    return deterministic_parse(user_text, task_map)


class OllamaRoomAgent:
    def __init__(
        self,
        model="qwen2.5:7b",
        url=DEFAULT_OLLAMA_URL,
        config_path="configs/living_room/tasks_v0.json",
        timeout=60,
    ):
        self.model = model
        self.url = url
        self.timeout = timeout
        self.cfg, self.tasks, self.task_map = load_tasks(config_path)
        self.system_prompt = build_system_prompt(self.tasks)
        self.history = [{"role": "system", "content": self.system_prompt}]

    def plan(self, user_text):
        messages = self.history + [{"role": "user", "content": user_text}]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": COMMAND_SCHEMA,
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
            },
        }

        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", "")
            plan = try_parse_json(content)
        except Exception as e:
            plan = {
                "action_type": "ask_clarification",
                "task_id": "",
                "object": "",
                "target": "",
                "reply": f"Ollama call failed: {e}",
                "safety_note": "No robot action executed.",
            }

        plan = validate_plan(plan, self.task_map, user_text)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": json.dumps(plan)})

        return plan
