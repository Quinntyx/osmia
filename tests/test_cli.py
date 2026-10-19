from __future__ import annotations

import json
from pathlib import Path

from osmia.cli import main


def test_models_and_config_cli(tmp_path, capsys) -> None:
    path = tmp_path / "config.json"
    assert main(["models", "--config", str(path)]) == 0
    output = capsys.readouterr().out
    assert "gpt-6.1-sol" in output
    assert "* gpt-6.1-sol" in output
    assert "deepseek-v4.1-flash-1017" in output
    assert "glm-5.3-flash" in output

    assert main(["config", "--config", str(path), "--json"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["providers"]["codex"]["access_token"] == "<unset>"

    assert main(["config", "--config", str(path)]) == 0
    text = capsys.readouterr().out
    assert "Config:" in text
    assert "not logged in" in text


def test_logout_and_login_key_cli(tmp_path, capsys, monkeypatch) -> None:
    path = tmp_path / "config.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert main(["login", "openrouter", "--config", str(path)]) == 0
    assert "OpenRouter credentials saved" in capsys.readouterr().out

    assert main(["logout", "openrouter", "--config", str(path)]) == 0
    assert "Cleared openrouter" in capsys.readouterr().out

    assert main(["login", "openrouter", "--config", str(path), "--api-key", ""]) == 1


def test_agents_cli_without_proxy_reports_error(capsys) -> None:
    code = main(["agents", "--proxy", "http://127.0.0.1:1"])
    assert code == 1
    assert "Cannot reach the Osmia proxy" in capsys.readouterr().err
