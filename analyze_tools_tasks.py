#!/usr/bin/env python3
"""
analyze_tools_tasks.py — Claw-Eval Tool & Task Classification Analysis

Scans all task.yaml files, extracts unique tools, then uses a local LLM
(sglang OpenAI-compatible API) to:
  1. Classify every unique tool as read-only vs. write/modify
  2. Classify every task into one of four workflow patterns

Usage:
    python analyze_tools_tasks.py [--base-url URL] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
TASKS_DIR = SCRIPT_DIR / "tasks"

SANDBOX_TOOLS_META = [
    {
        "name": "sandbox_shell_exec",
        "description": "Execute a shell command inside the sandbox container and return stdout/stderr. Can run arbitrary bash commands including file manipulation, compilation, database queries, etc.",
    },
    {
        "name": "sandbox_file_read",
        "description": "Read the contents of a file at the given path inside the sandbox container.",
    },
    {
        "name": "sandbox_file_write",
        "description": "Write content to a file at the given path inside the sandbox container (creates parent directories if needed).",
    },
    {
        "name": "sandbox_browser_screenshot",
        "description": "Capture screenshots of a web page inside the sandbox container's headless browser.",
    },
    {
        "name": "sandbox_read_media",
        "description": "Read and preview a media file (image, video, or PDF) inside the sandbox container. Returns metadata and base64-encoded frame images.",
    },
    {
        "name": "sandbox_pdf2image",
        "description": "Render PDF pages as images inside the sandbox container.",
    },
    {
        "name": "sandbox_file_download",
        "description": "Download a file from the sandbox container as binary (base64-encoded).",
    },
]

# ─── LLM helpers ─────────────────────────────────────────────────────


def get_model_name(base_url: str) -> str:
    resp = requests.get(f"{base_url}/models", timeout=15)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks that Qwen3 may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def llm_chat(
    base_url: str,
    model: str,
    user_prompt: str,
    *,
    max_tokens: int = 200,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise classifier. Respond ONLY with the "
                    "exact label requested — no explanation, no reasoning."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions", json=payload, timeout=120
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return _strip_thinking(text)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"    [retry {attempt+1}] {exc}")
            time.sleep(2 ** attempt)
    return ""


# ─── Classification prompts ─────────────────────────────────────────


def classify_tool(base_url: str, model: str, tool: dict) -> str:
    prompt = f"""\
Classify this tool as EXACTLY one of: "read-only" or "write/modify".

Definitions:
- "read-only": only retrieves / queries / searches / reads / observes data. Does NOT create, modify, delete, send, or write anything.
- "write/modify": creates, updates, deletes, sends, writes, or modifies data in the real world or external systems.

Tool name: {tool["name"]}
Tool description: {tool["description"]}

Answer with EXACTLY one of: read-only, write/modify"""

    raw = llm_chat(base_url, model, prompt, max_tokens=30)
    return _parse_tool_class(raw)


def _parse_tool_class(text: str) -> str:
    t = text.lower()
    if "read-only" in t or "read only" in t or "readonly" in t:
        return "read-only"
    if "write" in t or "modify" in t:
        return "write/modify"
    return "write/modify"


def classify_task(
    base_url: str,
    model: str,
    task: dict,
    tool_cls: dict[str, str],
) -> str:
    read_tools = [
        t for t in task["effective_tools"] if tool_cls.get(t) == "read-only"
    ]
    write_tools = [
        t for t in task["effective_tools"] if tool_cls.get(t) == "write/modify"
    ]

    prompt_text = (task["prompt"] or "")[:2000]
    ref_sol = (task["reference_solution"] or "")[:2000]

    prompt = f"""\
Classify this agent task into EXACTLY one of these four workflow patterns:

1. "Observation-only" — The agent only reads/queries/searches/analyzes data. No modifications to external systems at all.
2. "Observation-first" — The agent first systematically gathers information (reads/searches/analyzes), THEN performs ordered modifications. The observation phase is mostly completed before modifications begin.
3. "Mixed" — Observation and modification are interleaved throughout the task. The agent must alternate: observe → modify → observe again (because the modification changes what it will observe next).
4. "Modification-only" — The agent only performs modifications (create/write/send/delete) without needing to observe first.

Context for classification:
- Task: {task["task_id"]} — {task["task_name"]}
- Prompt: {prompt_text}
- Read-only tools available: {read_tools}
- Write/modify tools available: {write_tools}
- Reference solution: {ref_sol}

Answer with EXACTLY one of: Observation-only, Observation-first, Mixed, Modification-only"""

    raw = llm_chat(base_url, model, prompt, max_tokens=30)
    return _parse_task_class(raw)


def _parse_task_class(text: str) -> str:
    t = text.lower()
    ordered = [
        ("Observation-only", ["observation-only", "observation only"]),
        ("Observation-first", ["observation-first", "observation first"]),
        ("Modification-only", ["modification-only", "modification only"]),
        ("Mixed", ["mixed"]),
    ]
    for label, patterns in ordered:
        for p in patterns:
            if p in t:
                return label
    return "Mixed"


# ─── Data collection ─────────────────────────────────────────────────


def collect_all() -> tuple[dict[str, dict], list[dict]]:
    unique_tools: dict[str, dict] = {}
    tasks_data: list[dict] = []

    for yf in sorted(TASKS_DIR.glob("*/task.yaml")):
        with open(yf) as f:
            task = yaml.safe_load(f)

        task_id = task.get("task_id", yf.parent.name)
        tools_list = task.get("tools") or []
        explicit_names = [t["name"] for t in tools_list]
        uses_sandbox_only = len(tools_list) == 0

        effective = list(explicit_names)
        for st in SANDBOX_TOOLS_META:
            if st["name"] not in effective:
                effective.append(st["name"])

        tasks_data.append(
            {
                "task_id": task_id,
                "task_name": task.get("task_name", ""),
                "category": task.get("category", ""),
                "difficulty": task.get("difficulty", ""),
                "prompt": task.get("prompt", {}).get("text", ""),
                "explicit_tools": explicit_names,
                "effective_tools": effective,
                "uses_sandbox_only": uses_sandbox_only,
                "reference_solution": task.get("reference_solution", ""),
            }
        )

        for t in tools_list:
            name = t["name"]
            if name not in unique_tools:
                unique_tools[name] = {
                    "name": name,
                    "description": t.get("description", ""),
                    "source": "task-specific",
                }

    for st in SANDBOX_TOOLS_META:
        if st["name"] not in unique_tools:
            unique_tools[st["name"]] = {
                "name": st["name"],
                "description": st["description"],
                "source": "sandbox (implicit)",
            }

    return unique_tools, tasks_data


# ─── Report generation ───────────────────────────────────────────────


def generate_report(
    unique_tools: dict[str, dict],
    tasks_data: list[dict],
    tool_cls: dict[str, str],
    task_cls: dict[str, str],
    model_name: str,
) -> str:
    lines: list[str] = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("  Claw-Eval Tool & Task Classification Analysis Report")
    lines.append(sep)
    lines.append(f"Date        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model       : {model_name}")
    lines.append(f"Total tasks : {len(tasks_data)}")
    lines.append(f"Unique tools: {len(unique_tools)}")
    lines.append("")

    # ── Part 1: Tool classification ──
    lines.append(sep)
    lines.append("  PART 1: Tool Classification (read-only vs. write/modify)")
    lines.append(sep)
    lines.append("")

    ro_tools = [(n, t) for n, t in sorted(unique_tools.items()) if tool_cls.get(n) == "read-only"]
    wm_tools = [(n, t) for n, t in sorted(unique_tools.items()) if tool_cls.get(n) == "write/modify"]

    lines.append(f"Read-only tools ({len(ro_tools)}):")
    lines.append("-" * 60)
    for name, info in ro_tools:
        src = info.get("source", "")
        lines.append(f"  {name:40s} [{src}]")
        lines.append(f"    Description: {info['description'][:120]}")
    lines.append("")

    lines.append(f"Write/modify tools ({len(wm_tools)}):")
    lines.append("-" * 60)
    for name, info in wm_tools:
        src = info.get("source", "")
        lines.append(f"  {name:40s} [{src}]")
        lines.append(f"    Description: {info['description'][:120]}")
    lines.append("")

    total = len(unique_tools)
    lines.append("Tool Statistics:")
    lines.append(f"  Read-only   : {len(ro_tools):3d} / {total} ({len(ro_tools)/total*100:.1f}%)")
    lines.append(f"  Write/modify: {len(wm_tools):3d} / {total} ({len(wm_tools)/total*100:.1f}%)")
    lines.append("")

    # ── Part 2: Task classification ──
    lines.append(sep)
    lines.append("  PART 2: Task Classification (workflow pattern)")
    lines.append(sep)
    lines.append("")

    categories = ["Observation-only", "Observation-first", "Mixed", "Modification-only"]
    cat_tasks: dict[str, list[dict]] = {c: [] for c in categories}
    for t in tasks_data:
        c = task_cls.get(t["task_id"], "Mixed")
        cat_tasks[c].append(t)

    for cat in categories:
        items = cat_tasks[cat]
        lines.append(f"{cat} ({len(items)} tasks):")
        lines.append("-" * 60)
        for t in items:
            expl = t["explicit_tools"]
            tool_summary = ", ".join(expl[:5])
            if len(expl) > 5:
                tool_summary += f" ... (+{len(expl)-5} more)"
            if t["uses_sandbox_only"]:
                tool_summary = "[sandbox tools only]"
            lines.append(f"  {t['task_id']:45s}  tools: {tool_summary}")
        lines.append("")

    lines.append("Task Statistics:")
    total_tasks = len(tasks_data)
    for cat in categories:
        cnt = len(cat_tasks[cat])
        pct = cnt / total_tasks * 100 if total_tasks else 0
        lines.append(f"  {cat:20s}: {cnt:3d} / {total_tasks} ({pct:.1f}%)")
    lines.append("")

    # ── Part 3: Per-task detail ──
    lines.append(sep)
    lines.append("  PART 3: Per-Task Detail")
    lines.append(sep)
    lines.append("")
    for t in tasks_data:
        tc = task_cls.get(t["task_id"], "Unknown")
        lines.append(f"Task: {t['task_id']}")
        lines.append(f"  Name      : {t['task_name']}")
        lines.append(f"  Category  : {t['category']}")
        lines.append(f"  Difficulty: {t['difficulty']}")
        lines.append(f"  Pattern   : {tc}")
        ro = [n for n in t["explicit_tools"] if tool_cls.get(n) == "read-only"]
        wm = [n for n in t["explicit_tools"] if tool_cls.get(n) == "write/modify"]
        lines.append(f"  Read-only tools : {ro}")
        lines.append(f"  Write/mod tools : {wm}")
        if t["uses_sandbox_only"]:
            lines.append(f"  Note: uses sandbox tools only (no task-specific tools)")
        lines.append("")

    lines.append(sep)
    lines.append("  END OF REPORT")
    lines.append(sep)
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify claw-eval tools & tasks using a local LLM"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:30000/v1",
        help="sglang OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output log file path (default: analysis_result_<timestamp>.log)",
    )
    args = parser.parse_args()

    base_url = args.base_url
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(SCRIPT_DIR / f"analysis_result_{ts}.log")

    # Step 1: connect to LLM
    print(f"[1/5] Connecting to LLM at {base_url} ...")
    try:
        model = get_model_name(base_url)
    except Exception as exc:
        print(f"ERROR: Cannot connect to LLM — {exc}")
        print("Make sure sglang is running (bash run_sglang.sh)")
        sys.exit(1)
    print(f"  Model: {model}")

    # Step 2: collect data
    print(f"[2/5] Scanning tasks in {TASKS_DIR} ...")
    unique_tools, tasks_data = collect_all()
    print(f"  Found {len(unique_tools)} unique tools across {len(tasks_data)} tasks")

    # Step 3: classify tools
    print(f"[3/5] Classifying {len(unique_tools)} tools ...")
    tool_cls: dict[str, str] = {}
    for i, (name, info) in enumerate(sorted(unique_tools.items()), 1):
        cls = classify_tool(base_url, model, info)
        tool_cls[name] = cls
        print(f"  [{i:3d}/{len(unique_tools)}] {name:40s} => {cls}")

    ro_count = sum(1 for v in tool_cls.values() if v == "read-only")
    wm_count = sum(1 for v in tool_cls.values() if v == "write/modify")
    print(f"  Summary: {ro_count} read-only, {wm_count} write/modify")

    # Step 4: classify tasks
    print(f"[4/5] Classifying {len(tasks_data)} tasks ...")
    task_cls_map: dict[str, str] = {}
    for i, task in enumerate(tasks_data, 1):
        cls = classify_task(base_url, model, task, tool_cls)
        task_cls_map[task["task_id"]] = cls
        print(f"  [{i:3d}/{len(tasks_data)}] {task['task_id']:45s} => {cls}")

    # Step 5: generate report
    print(f"[5/5] Writing report to {output_path} ...")
    report = generate_report(unique_tools, tasks_data, tool_cls, task_cls_map, model)
    Path(output_path).write_text(report, encoding="utf-8")

    # Also save structured JSON for programmatic use
    json_path = output_path.replace(".log", ".json")
    structured = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "model": model,
            "total_tools": len(unique_tools),
            "total_tasks": len(tasks_data),
        },
        "tool_classifications": tool_cls,
        "task_classifications": task_cls_map,
        "tool_details": {
            name: {**info, "classification": tool_cls.get(name, "")}
            for name, info in unique_tools.items()
        },
        "task_details": [
            {**t, "classification": task_cls_map.get(t["task_id"], "")}
            for t in tasks_data
        ],
    }
    Path(json_path).write_text(
        json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Print summary to stdout
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total_t = len(unique_tools)
    print(f"Tools ({total_t} unique):")
    print(f"  Read-only    : {ro_count:3d} ({ro_count/total_t*100:.1f}%)")
    print(f"  Write/modify : {wm_count:3d} ({wm_count/total_t*100:.1f}%)")
    print()
    total_tasks = len(tasks_data)
    cats = ["Observation-only", "Observation-first", "Mixed", "Modification-only"]
    print(f"Tasks ({total_tasks} total):")
    for c in cats:
        cnt = sum(1 for v in task_cls_map.values() if v == c)
        print(f"  {c:20s}: {cnt:3d} ({cnt/total_tasks*100:.1f}%)")
    print()
    print(f"Report : {output_path}")
    print(f"JSON   : {json_path}")
    print("Done!")


if __name__ == "__main__":
    main()
