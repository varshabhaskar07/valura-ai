"""Smoke test: package imports cleanly and config loads with no env."""

from __future__ import annotations

import valura_ai
from valura_ai.config import Settings, get_settings


def test_package_version_exposed() -> None:
    assert isinstance(valura_ai.__version__, str)
    assert valura_ai.__version__.count(".") >= 2


def test_settings_defaults_with_no_env() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.model == "gpt-4o-mini"
    assert s.openai_api_key == ""
    assert s.request_timeout_s > s.classifier_timeout_s
    assert s.history_turns >= 1


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
