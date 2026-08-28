"""WebUI model picker must share Telegram /model's catalog.

Telegram ``/model`` and HUD Control already call ``list_picker_providers``.
The dashboard used to go through ``list_authenticated_providers`` plus
``include_unconfigured`` skeletons, which resurrected leftover slugs
(``opencode-free``, ``ox-alpha``) after they were excluded from Telegram.

These tests pin the shared-catalog contract — not a snapshot of today's
provider names.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.inventory import (
    ConfigContext,
    build_model_options_payload,
    load_picker_context,
)
from hermes_cli.model_switch import (
    list_authenticated_providers,
    list_picker_providers,
    picker_slug_is_excluded,
)


@pytest.fixture
def picker_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    return home


def _write_config(home: Path, **top_level) -> None:
    cfg = {
        "model": {"provider": "openrouter", "default": "openai/gpt-5.4"},
        "custom_providers": [],
    }
    cfg.update(top_level)
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _offline_discovery(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: {})
    monkeypatch.setattr(
        "hermes_cli.models.fetch_openrouter_models",
        lambda *a, **k: [("openai/gpt-5.4", "GPT")],
    )
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids",
        lambda *a, **k: ["openai/gpt-5.4"],
    )


def test_picker_slug_is_excluded_matches_alias():
    assert picker_slug_is_excluded("opencode-free", ["free"]) is True
    assert picker_slug_is_excluded("free", ["opencode-free"]) is True
    assert picker_slug_is_excluded("openrouter", ["opencode-free"]) is False


def test_leftover_user_config_excluded_hidden(picker_home, monkeypatch):
    """Leftover ``providers:`` rows must vanish when excluded — WebUI and Telegram."""
    _offline_discovery(monkeypatch)
    _write_config(
        picker_home,
        **{
            "model_catalog": {"excluded_providers": ["opencode-free", "ox-alpha"]},
            "providers": {
                "opencode-free": {"models": {"x-preview-f-free": {}}},
                "ox-alpha": {
                    "name": "OX Alpha",
                    "models": {"ox-alpha": {}},
                    "base_url": "https://leftover.example/v1",
                },
            },
        },
    )

    ctx = load_picker_context()
    telegram = list_picker_providers(
        current_provider=ctx.current_provider,
        current_base_url=ctx.current_base_url,
        current_model=ctx.current_model,
        user_providers=ctx.user_providers,
        custom_providers=ctx.custom_providers,
        excluded_providers=ctx.excluded_providers,
        include_moa=True,
        probe_custom_providers=False,
    )
    webui = build_model_options_payload(ctx)["providers"]

    for rows, label in ((telegram, "telegram"), (webui, "webui")):
        slugs = {str(r.get("slug", "")).lower() for r in rows}
        assert "opencode-free" not in slugs, f"{label} still lists opencode-free"
        assert "ox-alpha" not in slugs, f"{label} still lists ox-alpha"


def test_leftover_key_missing_user_config_hidden(picker_home, monkeypatch):
    """A leftover providers: row with no key and no endpoint is not a picker row."""
    _offline_discovery(monkeypatch)
    _write_config(
        picker_home,
        **{
            "providers": {
                "ox-alpha": {"name": "OX Alpha", "models": {"ox-alpha": {}}},
            },
        },
    )

    rows = list_authenticated_providers(
        current_provider="openrouter",
        user_providers={"ox-alpha": {"name": "OX Alpha", "models": {"ox-alpha": {}}}},
        custom_providers=[],
        for_picker=True,
        probe_custom_providers=False,
    )
    assert not any(str(r.get("slug", "")).lower() == "ox-alpha" for r in rows)


def test_webui_options_uses_list_picker_providers(monkeypatch):
    """Dashboard /api/model/options must call the Telegram picker function."""
    seen = {}

    def _fake_picker(**kwargs):
        seen.update(kwargs)
        return [
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": ["openai/gpt-5.4"],
                "total_models": 1,
                "is_current": True,
                "is_user_defined": False,
                "source": "built-in",
            }
        ]

    ctx = ConfigContext(
        current_provider="openrouter",
        current_model="openai/gpt-5.4",
        current_base_url="",
        user_providers={},
        custom_providers=[],
        excluded_providers=["opencode-free"],
    )
    with (
        patch("hermes_cli.model_switch.list_picker_providers", _fake_picker),
        patch(
            "hermes_cli.inventory._apply_pricing",
            lambda *a, **k: None,
        ),
        patch(
            "hermes_cli.inventory._apply_capabilities",
            lambda *a, **k: None,
        ),
        patch(
            "hermes_cli.inventory._apply_featured",
            lambda *a, **k: None,
        ),
    ):
        payload = build_model_options_payload(ctx, refresh=True)

    assert seen.get("excluded_providers") == ["opencode-free"]
    assert seen.get("refresh") is True
    assert {r["slug"] for r in payload["providers"]} >= {"openrouter"}


def test_webui_provider_slugs_match_telegram(picker_home, monkeypatch):
    """Same Hermes home → WebUI provider slugs === Telegram /model slugs."""
    _offline_discovery(monkeypatch)
    _write_config(
        picker_home,
        **{
            "model_catalog": {"excluded_providers": ["opencode-free"]},
            "providers": {
                "opencode-free": {"models": {"x-preview-f-free": {}}},
                "ox-alpha": {"name": "OX Alpha", "models": {"ox-alpha": {}}},
                "local-llama": {
                    "name": "Local Llama",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "models": {"llama-local": {}},
                    "discover_models": False,
                },
            },
        },
    )

    ctx = load_picker_context()
    telegram = list_picker_providers(
        current_provider=ctx.current_provider,
        current_base_url=ctx.current_base_url,
        current_model=ctx.current_model,
        user_providers=ctx.user_providers,
        custom_providers=ctx.custom_providers,
        excluded_providers=ctx.excluded_providers,
        include_moa=True,
        probe_custom_providers=False,
        probe_current_custom_provider=True,
    )
    webui = build_model_options_payload(ctx)["providers"]

    assert {r["slug"] for r in telegram} == {r["slug"] for r in webui}
    slugs = {r["slug"] for r in webui}
    assert "opencode-free" not in slugs
    assert "ox-alpha" not in slugs
    assert "local-llama" in slugs
