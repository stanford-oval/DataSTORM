#!/usr/bin/env python3
"""Push originality-eval prompts to Langfuse.

Adapted from an internal prompt-management helper.
Reads credentials via the same load_eval_credentials() shim that the codex
baseline uses, so it works on this host without a local secrets.toml.

Usage:
    python manage_langfuse_prompts.py push prompts/originality_pairwise_judge.json \
        --use-file-labels --commit-message "First push"
    python manage_langfuse_prompts.py push-all
    python manage_langfuse_prompts.py list --tag originality
    python manage_langfuse_prompts.py promote originality_pairwise_judge --version 2 --labels production
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from coding_agent_baselines.codex_baseline.run_acled_evaluations_codex import load_eval_credentials  # noqa: E402

load_eval_credentials()

from langfuse import get_client  # noqa: E402


def get_langfuse_client():
    return get_client()


def load_prompt_spec(path: Path) -> dict[str, Any]:
    raw = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level object")
    if "name" not in data:
        raise ValueError(f"{path} is missing required field: name")
    if "prompt" not in data:
        raise ValueError(f"{path} is missing required field: prompt")

    prompt_type = data.get("type")
    if prompt_type not in {None, "text", "chat"}:
        raise ValueError(f"{path} has invalid type={prompt_type!r}; expected 'text' or 'chat'")

    if prompt_type is None:
        data["type"] = "chat" if isinstance(data["prompt"], list) else "text"

    if data["type"] == "text" and not isinstance(data["prompt"], str):
        raise ValueError("Text prompts must use a string value for 'prompt'")
    if data["type"] == "chat" and not isinstance(data["prompt"], list):
        raise ValueError("Chat prompts must use a list value for 'prompt'")

    return data


def serialize_prompt(prompt_obj: Any) -> dict[str, Any]:
    return {
        "name": getattr(prompt_obj, "name", None),
        "version": getattr(prompt_obj, "version", None),
        "labels": list(getattr(prompt_obj, "labels", []) or []),
        "tags": list(getattr(prompt_obj, "tags", []) or []),
    }


def cmd_push(args: argparse.Namespace) -> int:
    spec = load_prompt_spec(args.file)
    client = get_langfuse_client()

    labels = args.labels
    if labels is None and args.use_file_labels:
        labels = list(spec.get("labels", []) or [])
    if labels is None:
        labels = []

    tags = args.tags if args.tags is not None else list(spec.get("tags", []) or [])
    config = spec.get("config", {}) or {}
    commit_message = args.commit_message if args.commit_message is not None else spec.get("commit_message")

    create_kwargs = {
        "name": args.name or spec["name"],
        "prompt": spec["prompt"],
        "labels": labels,
        "tags": tags,
        "type": spec["type"],
        "config": config,
        "commit_message": commit_message,
    }

    if args.dry_run:
        print(json.dumps(create_kwargs, indent=2, ensure_ascii=False)[:1500])
        return 0

    created = client.create_prompt(**create_kwargs)
    info = serialize_prompt(created)
    print(f"Created {info['name']} v{info['version']}  labels={info['labels']}  tags={info['tags']}")
    return 0


def cmd_push_all(args: argparse.Namespace) -> int:
    prompts_dir = args.dir or (REPO_ROOT / "prompts")
    files = sorted(prompts_dir.glob("*.json")) + sorted(prompts_dir.glob("*.yaml"))
    if not files:
        print(f"No prompt files under {prompts_dir}", file=sys.stderr)
        return 1
    rc = 0
    for f in files:
        ns = argparse.Namespace(
            file=f,
            name=None,
            labels=None,
            use_file_labels=True,
            tags=None,
            commit_message=args.commit_message,
            dry_run=args.dry_run,
        )
        try:
            cmd_push(ns)
        except Exception as exc:
            print(f"FAILED {f.name}: {exc}", file=sys.stderr)
            rc = 1
    return rc


def cmd_list(args: argparse.Namespace) -> int:
    client = get_langfuse_client()
    response = client.api.prompts.list(
        name=args.name, label=args.label, tag=args.tag, page=args.page, limit=args.limit
    )
    if not response.data:
        print("No prompts found.")
        return 0
    for item in response.data:
        labels = ",".join(getattr(item, "labels", []) or [])
        tags = ",".join(getattr(item, "tags", []) or [])
        print(f"{item.name:<40}  labels={labels:<25}  tags={tags}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    client = get_langfuse_client()
    if args.dry_run:
        print(json.dumps({"name": args.name, "version": args.version, "new_labels": args.labels}, indent=2))
        return 0
    client.update_prompt(name=args.name, version=args.version, new_labels=args.labels)
    print(f"Promoted {args.name} v{args.version} -> {args.labels}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Push and manage originality-eval prompts in Langfuse")
    sub = parser.add_subparsers(dest="command", required=True)

    p_push = sub.add_parser("push", help="Push a single prompt YAML")
    p_push.add_argument("file", type=Path)
    p_push.add_argument("--name")
    p_push.add_argument("--labels", nargs="+")
    p_push.add_argument("--use-file-labels", action="store_true")
    p_push.add_argument("--tags", nargs="+")
    p_push.add_argument("--commit-message")
    p_push.add_argument("--dry-run", action="store_true")
    p_push.set_defaults(func=cmd_push)

    p_all = sub.add_parser("push-all", help="Push every prompt in prompts/")
    p_all.add_argument("--dir", type=Path, default=None)
    p_all.add_argument("--commit-message")
    p_all.add_argument("--dry-run", action="store_true")
    p_all.set_defaults(func=cmd_push_all)

    p_list = sub.add_parser("list", help="List prompts")
    p_list.add_argument("--name")
    p_list.add_argument("--label")
    p_list.add_argument("--tag")
    p_list.add_argument("--page", type=int, default=1)
    p_list.add_argument("--limit", type=int, default=50)
    p_list.set_defaults(func=cmd_list)

    p_prom = sub.add_parser("promote", help="Move labels to a specific version")
    p_prom.add_argument("name")
    p_prom.add_argument("--version", type=int, required=True)
    p_prom.add_argument("--labels", nargs="+", required=True)
    p_prom.add_argument("--dry-run", action="store_true")
    p_prom.set_defaults(func=cmd_promote)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
