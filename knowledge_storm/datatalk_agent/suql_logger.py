"""Per-run logging of SUQL query executions.

Captures every ``answer(...)`` / ``summary(...)`` query that goes through
``sql_utils.execute_sql`` along with its outcome (ok / error / timeout)
and latency. The intended use is to feed real-world failures back to the
SUQL compiler for improvement.

Usage:
    from knowledge_storm.datatalk_agent import suql_logger
    suql_logger.set_log_path("/path/to/run/suql_executions.jsonl")
    # ... SUQL queries run ...

The logger is process-global and protected by a lock — the tree-search
fan-out runs DatatalkRM.forward() in worker threads, so contextvars
wouldn't propagate. The path is set once per run by the engine.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Optional

_log_path: Optional[str] = None
_lock = threading.Lock()


def set_log_path(path: Optional[str]) -> None:
    """Configure the JSONL file to append SUQL execution records to.

    Pass None to disable logging for the current process.
    """
    global _log_path
    with _lock:
        _log_path = path


def get_log_path() -> Optional[str]:
    return _log_path


def _looks_like_suql(sql: str) -> bool:
    """Cheap heuristic — log only queries that exercise SUQL UDFs."""
    lower = sql.lower()
    return "answer(" in lower or "summary(" in lower


def log_suql_execution(
    *,
    sql: str,
    status: str,
    error: Optional[str] = None,
    duration_ms: float = 0.0,
    database: Optional[str] = None,
) -> None:
    """Append one execution record to the configured log path.

    Silently no-ops when no path is configured or when the query is plain
    SQL (no SUQL UDFs). All errors are swallowed — logging must never
    affect the live pipeline.
    """
    path = _log_path
    if not path or not _looks_like_suql(sql):
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "duration_ms": round(duration_ms, 1),
        "database": database,
        "error": error,
        "sql": sql,
    }
    try:
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Best effort — never let logging break the live pipeline.
        pass
