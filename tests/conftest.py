"""Shared pytest fixtures.

Globally guarantees tests cannot reach the network: no OPENAI_API_KEY,
and yfinance is asked to use a local cache directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"


@pytest.fixture(autouse=True)
def _no_network_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI must run without an API key — enforce it at the fixture level."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("VALURA_MODEL", "gpt-4o-mini")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
