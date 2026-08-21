"""Tests for the deferred-report turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.deferred_report_stop import (
    assistant_promised_deferred_report,
    build_deferred_report_stop_nudge,
    deferred_report_stop_nudge_enabled,
    session_has_async_delivery,
    user_requested_deferred_report,
)


@pytest.fixture
def clear_deferred_env(monkeypatch):
    monkeypatch.delenv("HERMES_DEFERRED_REPORT_STOP_NUDGE", raising=False)
    return monkeypatch


def test_enabled_by_default(clear_deferred_env):
    assert deferred_report_stop_nudge_enabled() is True


def test_env_can_disable(clear_deferred_env):
    clear_deferred_env.setenv("HERMES_DEFERRED_REPORT_STOP_NUDGE", "0")
    assert deferred_report_stop_nudge_enabled() is False
    assert (
        build_deferred_report_stop_nudge(
            messages=[{"role": "user", "content": "Report back when both land."}],
            final_assistant_text="OK, I'll report back when both land.",
        )
        is None
    )


def test_user_requested_report_back(clear_deferred_env):
    messages = [
        {"role": "user", "content": "how is redis going?"},
        {"role": "assistant", "content": "still investigating"},
        {"role": "user", "content": "Report back when both land."},
    ]
    assert user_requested_deferred_report(messages) is True


def test_assistant_promise_detection(clear_deferred_env):
    assert assistant_promised_deferred_report(
        [],
        final_assistant_text="I'll report back when both land.",
    )
    assert not assistant_promised_deferred_report(
        [],
        final_assistant_text="Redis fix is merged and verified.",
    )


def test_nudge_when_user_asks_without_async_delivery(clear_deferred_env):
    messages = [
        {"role": "user", "content": "Report back when both land."},
    ]
    nudge = build_deferred_report_stop_nudge(
        messages=messages,
        final_assistant_text="Got it — redirected current run.",
        available_tools=["cronjob", "terminal", "delegate_task"],
    )
    assert nudge is not None
    assert "cronjob" in nudge
    assert "notify_on_complete" in nudge
    assert "delegate_task" in nudge
    assert "verbal" in nudge.lower() or "promise" in nudge.lower()


def test_nudge_when_assistant_promises_without_user_phrase(clear_deferred_env):
    messages = [
        {"role": "user", "content": "keep going on 2 and 3"},
    ]
    nudge = build_deferred_report_stop_nudge(
        messages=messages,
        final_assistant_text="I'll update you when both land.",
        available_tools=["cronjob", "terminal"],
    )
    assert nudge is not None
    assert "cronjob" in nudge


def test_no_nudge_when_cronjob_already_created(clear_deferred_env):
    messages = [
        {"role": "user", "content": "Report back when both land."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "cronjob",
                        "arguments": (
                            '{"action":"create","schedule":"15m",'
                            '"prompt":"check status","deliver":"origin"}'
                        ),
                    },
                }
            ],
        },
    ]
    assert session_has_async_delivery(messages) is True
    assert (
        build_deferred_report_stop_nudge(
            messages=messages,
            final_assistant_text="Cron set — I'll report when both land.",
            available_tools=["cronjob", "terminal"],
        )
        is None
    )


def test_no_nudge_when_terminal_notify_scheduled(clear_deferred_env):
    messages = [
        {"role": "user", "content": "Notify me when deploy finishes."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": (
                            '{"command":"./deploy.sh","background":true,'
                            '"notify_on_complete":true}'
                        ),
                    },
                }
            ],
        },
    ]
    assert (
        build_deferred_report_stop_nudge(
            messages=messages,
            final_assistant_text="Deploy started; I'll report back when it lands.",
            available_tools=["terminal"],
        )
        is None
    )


def test_attempt_budget(clear_deferred_env):
    messages = [{"role": "user", "content": "Report back when both land."}]
    assert (
        build_deferred_report_stop_nudge(
            messages=messages,
            final_assistant_text="ok",
            attempts=2,
            available_tools=["cronjob"],
        )
        is None
    )


def test_no_nudge_without_scheduling_tools(clear_deferred_env):
    messages = [{"role": "user", "content": "Report back when both land."}]
    assert (
        build_deferred_report_stop_nudge(
            messages=messages,
            final_assistant_text="I'll report back when both land.",
            available_tools=["read_file", "web_search"],
        )
        is None
    )


def test_no_nudge_for_ordinary_status_reply(clear_deferred_env):
    messages = [
        {"role": "user", "content": "Status?"},
    ]
    assert (
        build_deferred_report_stop_nudge(
            messages=messages,
            final_assistant_text="Still working on Redis scoping.",
            available_tools=["cronjob", "terminal"],
        )
        is None
    )
