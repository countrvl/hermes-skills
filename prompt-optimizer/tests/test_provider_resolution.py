import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock


REPO_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZE_PATH = REPO_ROOT / "prompt-optimizer" / "scripts" / "optimize.py"


def load_optimize_module():
    spec = importlib.util.spec_from_file_location("prompt_optimizer_optimize", OPTIMIZE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_hermes_config(hermes_home: Path, *, provider: str, model: str) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        f"""
model:
  provider: {provider}
  default: {model}
""".lstrip(),
        encoding="utf-8",
    )


def write_auth_json(
    hermes_home: Path,
    *,
    provider: str,
    base_url: str,
    label: str,
) -> None:
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "credential_pool": {
                    provider: [
                        {
                            "label": label,
                            "base_url": base_url,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )


def configure_module(
    module,
    *,
    provider: str,
    api_base: str,
    api_key: str = "test-key",
    model: str = "test-model",
):
    module.PROVIDER = provider
    module.API_BASE = api_base
    module.API_KEY = api_key
    module.MODEL = model
    module.MAX_RETRIES = 0


class FakeAsyncClient:
    post = AsyncMock()

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def install_fake_httpx(monkeypatch, post):
    fake_httpx = types.SimpleNamespace(AsyncClient=FakeAsyncClient)
    FakeAsyncClient.post = post
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)


class StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_resolve_config_uses_hermes_anthropic_config_auth_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PROMPT_OPTIMIZER_API_KEY", raising=False)
    monkeypatch.delenv("PROMPT_OPTIMIZER_API_BASE", raising=False)
    monkeypatch.delenv("PROMPT_OPTIMIZER_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_API_KEY", raising=False)

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    write_hermes_config(
        hermes_home,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    write_auth_json(
        hermes_home,
        provider="anthropic",
        base_url="https://api.anthropic.com/v1/",
        label="ANTHROPIC_API_KEY",
    )
    (hermes_home / ".env").write_text(
        "ANTHROPIC_API_KEY=env-anthropic-key\n",
        encoding="utf-8",
    )

    module = load_optimize_module()

    api_key, base_url, model, provider = module._resolve_config()

    assert api_key == "env-anthropic-key"
    assert base_url == "https://api.anthropic.com/v1"
    assert model == "claude-sonnet-4-6"
    assert provider == "anthropic"


def test_call_llm_routes_anthropic_to_messages_adapter(monkeypatch):
    module = load_optimize_module()
    configure_module(
        module,
        provider="anthropic",
        api_base="https://api.anthropic.com/v1",
        api_key="anthropic-key",
        model="claude-sonnet-4-6",
    )
    post = AsyncMock(
        return_value=StubResponse({"content": [{"type": "text", "text": '{"ok": true}'}]})
    )
    install_fake_httpx(monkeypatch, post)

    result = asyncio.run(module.call_llm("system prompt", "user message"))

    assert result == {"ok": True}
    post.assert_awaited_once()
    url = post.await_args.args[0]
    headers = post.await_args.kwargs["headers"]
    payload = post.await_args.kwargs["json"]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "anthropic-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert payload["system"] == "system prompt"
    assert payload["messages"] == [{"role": "user", "content": "user message"}]
    assert all(message["role"] == "user" for message in payload["messages"])


def test_call_llm_routes_openai_to_chat_completions(monkeypatch):
    module = load_optimize_module()
    configure_module(module, provider="openai", api_base="https://api.openai.com/v1")
    post = AsyncMock(
        return_value=StubResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})
    )
    install_fake_httpx(monkeypatch, post)

    result = asyncio.run(module.call_llm("system prompt", "user message"))

    assert result == {"ok": True}
    post.assert_awaited_once()
    assert post.await_args.args[0] == "https://api.openai.com/v1/chat/completions"


def test_call_llm_routes_deepseek_to_chat_completions(monkeypatch):
    module = load_optimize_module()
    configure_module(module, provider="deepseek", api_base="https://api.deepseek.com/v1")
    post = AsyncMock(
        return_value=StubResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})
    )
    install_fake_httpx(monkeypatch, post)

    result = asyncio.run(module.call_llm("system prompt", "user message"))

    assert result == {"ok": True}
    post.assert_awaited_once()
    assert post.await_args.args[0] == "https://api.deepseek.com/v1/chat/completions"
