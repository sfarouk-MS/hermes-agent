"""Auto-background + notify for long deploy/rollout terminal commands.

Long foreground deploys block the chat turn and starve interim status.
Hermes promotes matching commands to background=true + notify_on_complete=true
so agents stay responsive and must verify after the completion ping.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.terminal_tool import (
    _autobackground_long_deploy,
    _looks_like_long_deploy,
)


@pytest.mark.parametrize(
    "command",
    [
        "fly deploy",
        "vercel deploy --prod",
        "npm run deploy",
        "pnpm deploy",
        "make deploy",
        "helm upgrade myapp ./chart",
        "kubectl apply -f manifests/",
        "kubectl rollout status deploy/api",
        "terraform apply -auto-approve",
        "pulumi up",
        "ansible-playbook site.yml",
        "./scripts/deploy-live.sh",
        "bash deploy.sh --env=prod",
        "python scripts/deploy.py",
        "docker stack deploy -c compose.yml stack",
        "gcloud run deploy api --image=gcr.io/x/api",
        "deploy",
    ],
)
def test_looks_like_long_deploy_positive(command):
    assert _looks_like_long_deploy(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "git status",
        "gh pr checks 999",
        "while true; do gh pr checks 999; sleep 30; done",
        "npm run test",
        "pytest -q",
        "fly deploy --help",
        "helm upgrade --version",
        'git commit -m "please deploy tomorrow"',
        "echo deploy",
        "cat deploy.md",
    ],
)
def test_looks_like_long_deploy_negative(command):
    assert _looks_like_long_deploy(command) is False


def test_autobackground_promotes_foreground_deploy():
    background, notify, note = _autobackground_long_deploy(
        command="fly deploy",
        background=False,
        notify_on_complete=False,
        watch_patterns=None,
    )
    assert background is True
    assert notify is True
    assert "Auto-backgrounded" in note
    assert "notify_on_complete" in note
    assert "post-deploy" in note.lower() or "verification" in note.lower()


def test_autobackground_promotes_notify_when_already_background():
    background, notify, note = _autobackground_long_deploy(
        command="./deploy.sh",
        background=True,
        notify_on_complete=False,
        watch_patterns=None,
    )
    assert background is True
    assert notify is True
    assert "notify_on_complete" in note
    assert "Auto-backgrounded" not in note


def test_autobackground_respects_watch_patterns():
    background, notify, note = _autobackground_long_deploy(
        command="helm upgrade app ./chart",
        background=True,
        notify_on_complete=False,
        watch_patterns=["RELEASE SUCCESSFUL"],
    )
    assert background is True
    assert notify is False
    assert note == ""


def test_autobackground_noop_for_non_deploy():
    background, notify, note = _autobackground_long_deploy(
        command="echo hello",
        background=False,
        notify_on_complete=False,
        watch_patterns=None,
    )
    assert background is False
    assert notify is False
    assert note == ""


def _harness(monkeypatch, tmp_path, *, capture_notify):
    """Patch terminal_tool enough to observe background spawn flags."""
    import tools.terminal_tool as terminal_tool_module
    from tools import process_registry as process_registry_module

    config = {
        "env_type": "local",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "cwd": str(tmp_path),
        "timeout": 30,
    }
    dummy_env = SimpleNamespace(env={})
    capture_notify["value"] = None

    def fake_spawn_local(**kwargs):
        session = SimpleNamespace(
            id="proc_deploy_test",
            pid=4242,
            notify_on_complete=False,
            watcher_platform="",
            watcher_chat_id="",
            watcher_user_id="",
            watcher_user_name="",
            watcher_thread_id="",
            watcher_message_id="",
            watcher_interval=0,
        )
        return session

    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setattr(process_registry_module.process_registry, "spawn_local", fake_spawn_local)
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)
    return terminal_tool_module


def test_terminal_tool_autobackgrounds_foreground_deploy(monkeypatch, tmp_path):
    capture = {}
    tt = _harness(monkeypatch, tmp_path, capture_notify=capture)
    try:
        result = json.loads(
            tt.terminal_tool(command="vercel deploy --prod", background=False)
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert result["session_id"] == "proc_deploy_test"
    assert result.get("notify_on_complete") is True
    assert "auto_background" in result
    assert "Auto-backgrounded" in result["auto_background"]
    # No silent-process hint — notify was promoted.
    assert "silent" not in result.get("hint", "").lower()


def test_terminal_tool_autocompletes_notify_on_bg_deploy(monkeypatch, tmp_path):
    capture = {}
    tt = _harness(monkeypatch, tmp_path, capture_notify=capture)
    try:
        result = json.loads(
            tt.terminal_tool(command="make deploy", background=True)
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert result.get("notify_on_complete") is True
    assert "notify_on_complete" in result.get("auto_background", "")
