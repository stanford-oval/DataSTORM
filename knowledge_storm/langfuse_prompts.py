"""Retrieve prompts from Langfuse, falling back to the snapshot in ``prompts/``.

Prompts are authored in Langfuse, which stays the source of truth for the
maintainers. For everyone else -- and for offline or reproducible runs -- the
same prompts are checked into ``prompts/`` as JSON by
``scripts/export_langfuse_prompts.py``, and this module loads them from there.

Selection is controlled by ``DATASTORM_PROMPT_SOURCE``:

``auto`` (default)
    Use Langfuse when credentials are configured and the fetch succeeds;
    otherwise fall back to the local snapshot.
``langfuse``
    Require Langfuse. Fetch failures raise.
``local``
    Ignore Langfuse entirely and read the snapshot. Use this to reproduce a
    published run without depending on the live prompt store.

Set ``DATASTORM_PROMPTS_DIR`` to point at a different snapshot directory.
"""

from __future__ import annotations

import json
import os
import re
from functools import cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape

from knowledge_storm.log_utils import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MUSTACHE = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


class LocalPrompt:
    """Minimal stand-in for a Langfuse prompt client, backed by a JSON file.

    Implements the surface the pipeline actually uses: ``.prompt`` (raw text or
    chat messages) and ``.compile()``. It is deliberately *not* a Langfuse
    object -- generation tracing links prompts by server-side ID, which a local
    snapshot has no equivalent for, so callers skip that linkage.
    """

    is_local = True

    def __init__(self, payload: dict[str, Any]):
        self.name: str = payload["name"]
        self.version = payload.get("version")
        self.type: str = payload.get("type", "text")
        self.labels: list[str] = payload.get("labels", [])
        self.config: dict[str, Any] = payload.get("config", {}) or {}
        self.prompt = payload["prompt"]

    @staticmethod
    def _substitute(template: str, variables: dict[str, Any]) -> str:
        def repl(match: re.Match) -> str:
            key = match.group(1)
            return str(variables[key]) if key in variables else match.group(0)

        return _MUSTACHE.sub(repl, template)

    def compile(self, **variables: Any):
        """Mirror Langfuse's compile(): str for text prompts, messages for chat."""
        if self.type == "chat" or isinstance(self.prompt, list):
            return [
                {
                    "role": msg["role"],
                    "content": self._substitute(msg["content"], variables),
                }
                for msg in self.prompt
            ]
        return self._substitute(self.prompt, variables)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LocalPrompt(name={self.name!r}, version={self.version})"


def _prompts_dir() -> Path:
    return Path(os.getenv("DATASTORM_PROMPTS_DIR", str(_REPO_ROOT / "prompts")))


@cache
def _get_langfuse_client():
    from langfuse import get_client

    return get_client()


def _langfuse_configured() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )


@cache
def _load_local(group: str) -> LocalPrompt:
    base = _prompts_dir()
    candidate = base / f"{group}.json"
    if candidate.exists():
        return LocalPrompt(json.loads(candidate.read_text(encoding="utf-8")))
    raise RuntimeError(
        f"Prompt {group!r} not found in Langfuse or in the local snapshot at "
        f"{base}. Run scripts/export_langfuse_prompts.py to regenerate the "
        f"snapshot, or set DATASTORM_PROMPTS_DIR."
    )


@cache
def _get_prompt(group: str):
    source = os.getenv("DATASTORM_PROMPT_SOURCE", "auto").lower()

    if source == "local":
        return _load_local(group)

    if source == "langfuse" or (source == "auto" and _langfuse_configured()):
        try:
            return _get_langfuse_client().get_prompt(group)
        except Exception as exc:
            if source == "langfuse":
                raise RuntimeError(
                    f"Could not fetch prompt {group!r} from Langfuse: {exc}"
                ) from exc
            logger.warning(
                f"Langfuse fetch failed for prompt {group!r} ({exc}); "
                f"using local snapshot"
            )

    return _load_local(group)


def compile_prompt(group: str, variables: dict[str, str]):
    """Fetch a prompt and compile it with ``variables``.

    Returns a tuple of the raw prompt object and the compiled chat messages
    as ``list[tuple[str, str]]`` ready for the LLM call.
    """

    prompt = _get_prompt(group)
    compiled = prompt.compile(**variables)
    messages: list[tuple[str, str]] = [(msg["role"], msg["content"]) for msg in compiled]
    return prompt, messages


def get_compiled_messages(group: str, variables: dict[str, str]) -> list[tuple[str, str]]:
    """Convenience wrapper returning only compiled messages."""
    _, msgs = compile_prompt(group, variables)
    return msgs


@cache
def _get_jinja_env():
    """Create a cached Jinja2 environment with safe defaults."""
    return Environment(
        autoescape=select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def compile_prompt_jinja(group: str, variables: dict[str, any]) -> tuple[any, list[tuple[str, str]]]:
    """Fetch prompt and compile it with full Jinja2 support.

    This enhanced version supports:
    - For loops: {% for item in items %}...{% endfor %}
    - Conditionals: {% if condition %}...{% endif %}
    - Filters: {{ value|upper }}
    - List operations: {{ items|length }}
    - And all other Jinja2 features

    Args:
        group: The prompt group identifier
        variables: Dictionary of variables to compile into the prompt

    Returns:
        A tuple of the raw prompt object and the compiled chat messages
        as ``list[tuple[str, str]]`` ready for the LLM call.
    """
    prompt = _get_prompt(group)

    # Get the prompt content
    prompt_content = prompt.prompt

    # Create Jinja2 environment
    env = _get_jinja_env()

    compiled_messages = []

    # Check if this is a text prompt (string) or chat prompt (list of messages)
    if isinstance(prompt_content, str):
        # This is a text prompt - treat it as a single user message
        template = env.from_string(prompt_content)
        compiled_content = template.render(**variables)
        compiled_messages.append(("user", compiled_content))
    else:
        # This is a chat prompt - process each message
        for message in prompt_content:
            role = message["role"]
            content_template = message["content"]

            # Create a template from the content string
            template = env.from_string(content_template)

            # Render the template with variables
            compiled_content = template.render(**variables)

            compiled_messages.append((role, compiled_content))

    return prompt, compiled_messages
