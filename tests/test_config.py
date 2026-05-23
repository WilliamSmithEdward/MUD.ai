from __future__ import annotations

import json
from pathlib import Path

from mudai.config import AppConfig, CONFIG_PATH


def test_load_creates_file_with_defaults(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake = tmp_path / "config.local.json"
    monkeypatch.setattr("mudai.config.CONFIG_PATH", fake)
    cfg = AppConfig.load()
    assert fake.exists()
    assert cfg.mud.host == "mud.arctic.org"
    assert cfg.mud.port == 2700
    assert cfg.llm.model_file.endswith(".gguf")


def test_roundtrip_save_load(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake = tmp_path / "config.local.json"
    monkeypatch.setattr("mudai.config.CONFIG_PATH", fake)
    cfg = AppConfig.load()
    cfg.llm.temperature = 0.123
    cfg.agent.auto_send = True
    cfg.save()
    data = json.loads(fake.read_text("utf-8"))
    assert data["llm"]["temperature"] == 0.123
    assert data["agent"]["auto_send"] is True
    cfg2 = AppConfig.load()
    assert cfg2.llm.temperature == 0.123
    assert cfg2.agent.auto_send is True


def test_models_catalog_default_resolvable() -> None:
    from mudai.llm import models_catalog
    entry = models_catalog.by_key(models_catalog.DEFAULT_KEY)
    assert entry is not None
    assert entry.filename.endswith(".gguf")


def test_models_catalog_lookup_by_filename() -> None:
    from mudai.llm import models_catalog
    entry = models_catalog.by_key(models_catalog.DEFAULT_KEY)
    assert entry is not None
    assert models_catalog.by_filename(entry.filename) is entry
    assert models_catalog.by_filename("nope.gguf") is None


def test_app_config_module_path_unchanged() -> None:
    # Ensure CONFIG_PATH constant is defined at module level (regression).
    assert CONFIG_PATH.name == "config.local.json"
