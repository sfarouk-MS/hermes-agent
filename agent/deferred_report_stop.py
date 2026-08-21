"""Turn-end guard against verbal \"I'll report back later\" exits.

Gateway agents (especially multi-topic Telegram dashboards) often acknowledge
\"report back when both land\" and stop with ``finish_reason=stop`` without
scheduling any async delivery. Other topics/agents never wake this session, so
the chat goes permanently silent.

This module is policy-only: when the model tries to finish on a deferred-report
promise without registering cron / notify_on_complete / background delegation,
return a bounded synthetic nudge so the loop continues and schedules delivery.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, Optional


_DEFAULT_MAX_ATTEMPTS = 2

# User asked to be woken later.
_USER_REPORT_BACK_RE = re.compile(
    r"(?i)\b("
    r"report\s+back\s+when|"
    r"notify\s+me\s+when|"
    r"ping\s+me\s+when|"
    r"update\s+me\s+when|"
    r"let\s+me\s+know\s+when|"
    r"keep\s+me\s+(?:posted|updated)|"
    r"when\s+both\s+land|"
    r"when\s+(?:it|they|that|this|both)\s+"
    r"(?:land|lands|done|finish(?:es|ed)?|complete[sd]?)"
    r")\b"
)

# Assistant ended on a promise of future notification / follow-up.
_ASSISTANT_PROMISE_RE = re.compile(
    r"(?i)\b("
    r"(?:i(?:['’]ll| will)|we(?:['’]ll| will)|i(?:['’]m| am)\s+going\s+to)\s+"
    r"(?:report|update|notify|ping|circle\s+back|follow\s+up)|"
    r"report\s+back\s+when|"
    r"notify\s+you\s+when|"
    r"ping\s+you\s+when|"
    r"update\s+you\s+when|"
    r"keep\s+you\s+(?:posted|updated)|"
    r"when\s+both\s+land|"
    r"once\s+(?:both|they|it)\s+"
    r"(?:land|lands|done|finish(?:es|ed)?|complete[sd]?)"
    r")\b"
)

_ASYNC_DELIVERY_TOOLS = frozenset(
    {
        "cronjob",
        "terminal",
        "delegate_task",
    }
)


def deferred_report_stop_nudge_enabled() -> bool:
    """Return whether the deferred-report stop-guard is active.

    On by default. Set ``HERMES_DEFERRED_REPORT_STOP_NUDGE=0`` to disable.
    """
    env = os.environ.get("HERMES_DEFERRED_REPORT_STOP_NUDGE")
    if env is not None and env.strip().lower() in {"0", "false", "no", "off"}:
        return False
    return True


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name") or "")
        return str(tc.get("name") or "")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return str(getattr(fn, "name", "") or "")
    return str(getattr(tc, "name", "") or "")


def _tool_call_args(tc: Any) -> dict:
    raw: Any = None
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            raw = fn.get("arguments")
        else:
            raw = tc.get("arguments")
    else:
        fn = getattr(tc, "function", None)
        if fn is not None:
            raw = getattr(fn, "arguments", None)
        else:
            raw = getattr(tc, "arguments", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _message_text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {None, "text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def user_requested_deferred_report(messages: Iterable[dict] | None) -> bool:
    """True when a recent user message asked to be notified later."""
    if not messages:
        return False
    # Scan newest-first; stop after a few user turns so old history can't
    # permanently re-arm the guard.
    seen_users = 0
    for msg in reversed(list(messages)):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        if msg.get("_deferred_report_stop_synthetic") or msg.get("_kanban_stop_synthetic"):
            continue
        seen_users += 1
        if _USER_REPORT_BACK_RE.search(_message_text(msg) or ""):
            return True
        if seen_users >= 3:
            break
    return False


def assistant_promised_deferred_report(
    messages: Iterable[dict] | None,
    *,
    final_assistant_text: str | None = None,
) -> bool:
    """True when the finishing assistant text promises a later report."""
    text = (final_assistant_text or "").strip()
    if not text and messages:
        for msg in reversed(list(messages)):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            text = _message_text(msg).strip()
            break
    if not text:
        return False
    return bool(_ASSISTANT_PROMISE_RE.search(text))


def session_has_async_delivery(messages: Iterable[dict] | None = None) -> bool:
    """True when something already registered a wake-up for this session."""
    try:
        from tools.process_registry import process_registry

        if getattr(process_registry, "pending_watchers", None):
            return True
        # Live bg sessions marked notify_on_complete also count — the gateway
        # may not have drained pending_watchers yet mid-turn.
        running = getattr(process_registry, "_running", None) or {}
        for sess in running.values():
            if getattr(sess, "notify_on_complete", False):
                return True
            if getattr(sess, "watch_patterns", None):
                return True
    except Exception:
        pass

    try:
        from tools.async_delegation import active_count

        if active_count() > 0:
            return True
    except Exception:
        pass

    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = _tool_call_name(tc)
            if name not in _ASYNC_DELIVERY_TOOLS:
                continue
            args = _tool_call_args(tc)
            if name == "cronjob" and str(args.get("action") or "").lower() == "create":
                return True
            if name == "delegate_task" and args.get("background") in (
                True,
                "true",
                "True",
                1,
                "1",
            ):
                return True
            if name == "terminal" and (
                args.get("notify_on_complete") in (True, "true", "True", 1, "1")
                or args.get("watch_patterns")
            ):
                return True
    return False


def build_deferred_report_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    final_assistant_text: str | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    available_tools: Iterable[str] | None = None,
) -> Optional[str]:
    """Return a synthetic follow-up when the model exits on a deferred promise.

    Returns ``None`` when the guard should not fire.
    """
    if not deferred_report_stop_nudge_enabled():
        return None
    if attempts >= max_attempts:
        return None

    user_asked = user_requested_deferred_report(messages)
    assistant_promised = assistant_promised_deferred_report(
        messages, final_assistant_text=final_assistant_text
    )
    if not (user_asked or assistant_promised):
        return None
    if session_has_async_delivery(messages):
        return None

    tools = {str(t) for t in (available_tools or []) if t}
    options: list[str] = []
    if not tools or "cronjob" in tools:
        options.append(
            "`cronjob(action='create', schedule='…', prompt='…', deliver='origin')` "
            "to poll/check and report back to this chat"
        )
    if not tools or "terminal" in tools:
        options.append(
            "`terminal(..., background=true, notify_on_complete=true)` for a "
            "bounded wait/poller that wakes this session on exit"
        )
    if not tools or "delegate_task" in tools:
        options.append(
            "`delegate_task(..., background=true)` so the child result re-enters "
            "this conversation when finished"
        )
    if not options:
        # No scheduling tools in the toolset — can't self-correct.
        return None

    numbered = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options, 1))
    return (
        "[System: You ended the turn on a deferred-report promise without "
        "scheduling any delivery. Other agents/topics do NOT notify this chat "
        "automatically — a verbal \"I'll report back when…\" leaves the user "
        "in permanent silence.\n\n"
        "Do this immediately in your next response — do not only narrate:\n"
        f"{numbered}\n\n"
        "Then confirm to the user what you scheduled and when they should "
        "expect the update. Never end again with only a promise.]"
    )


__all__ = [
    "assistant_promised_deferred_report",
    "build_deferred_report_stop_nudge",
    "deferred_report_stop_nudge_enabled",
    "session_has_async_delivery",
    "user_requested_deferred_report",
]
