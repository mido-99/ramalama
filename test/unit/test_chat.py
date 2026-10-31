import argparse
import http.client
import io
import json
import urllib.error
from unittest.mock import MagicMock

import pytest

import ramalama.chat as chat_module
from ramalama.chat import RamaLamaShell, wait_for_server_ready
from ramalama.chat_providers.openai import OpenAICompletionsChatProvider


def test_input_with_backslash():
    # fake arguments to not get errors
    mock_args = MagicMock()
    mock_args.prefix = "> "
    mock_args.url = "http://localhost:8080"
    mock_args.model = "test-model"
    mock_args.rag = None
    mock_args.mcp = []
    mock_args.summarize_after = 0
    mock_args.color = "never"

    # instance of RamaLamaShell with the fake args
    ramalama = RamaLamaShell(mock_args)

    # disable network function
    ramalama._req = MagicMock(return_value="Answer")

    # Case 1, with backslash
    assert ramalama.default("Hola\\") is False
    assert ramalama.content == ["Hola"]

    # Verify that _req was not called
    ramalama._req.assert_not_called()


def test_input_continuation_with_backslash():
    # fake arguments to not get errors
    mock_args = MagicMock()
    mock_args.prefix = "> "
    mock_args.url = "http://localhost:8080"
    mock_args.model = "test-model"
    mock_args.rag = None
    mock_args.mcp = []
    mock_args.summarize_after = 0
    mock_args.color = "never"

    # instance of RamaLamaShell with the fake args
    ramalama = RamaLamaShell(mock_args)
    ramalama._req = MagicMock(return_value="Answer")  # disable network function

    # case 2, we add text after backslash
    assert ramalama.default("Hola \\") is False
    assert ramalama.content == ["Hola "]

    # Verify that _req was not called
    ramalama._req.assert_not_called()

    # we add text after
    assert ramalama.default("Mundo") is not False
    assert ramalama.content == []

    # Verify that _req was called
    ramalama._req.assert_called_once()


def test_input_without_backslash():
    # fake arguments to not get errors
    mock_args = MagicMock()
    mock_args.prefix = "> "
    mock_args.url = "http://localhost:8080"
    mock_args.model = "test-model"
    mock_args.rag = None
    mock_args.mcp = []
    mock_args.summarize_after = 0
    mock_args.color = "never"

    # instance of RamaLamaShell with the fake args
    ramalama = RamaLamaShell(mock_args)
    ramalama._req = MagicMock(return_value="Answer")  # disable network function

    # case 3, no backslash
    result = ramalama.default("Hola")
    assert result is not False  # it does not return True
    assert ramalama.content == []  # the message has been sended

    # Verify that _req was called
    ramalama._req.assert_called_once()


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _FakeResponse:
    def __init__(self, body: bytes = b"{}") -> None:
        self.status = 200
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://test", code, "error", {}, io.BytesIO(body))


class _UrlOpener:
    """urlopen stand-in that replays scripted outcomes and records the calls."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, float]] = []

    def __call__(self, request, timeout=None):
        self.calls.append((request.full_url, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


def _models_body(owned_by: str) -> _FakeResponse:
    return _FakeResponse(json.dumps({"data": [{"id": "m", "owned_by": owned_by}]}).encode())


def _provider() -> OpenAICompletionsChatProvider:
    return OpenAICompletionsChatProvider("http://127.0.0.1:8080/v1")


def _prepare(monkeypatch, outcomes: list[object]) -> tuple[_FakeClock, _UrlOpener]:
    clock = _FakeClock()
    opener = _UrlOpener(outcomes)
    monkeypatch.setattr(chat_module, "time", clock)
    monkeypatch.setattr(chat_module, "Spinner", MagicMock())
    monkeypatch.setattr(chat_module.urllib.request, "urlopen", opener)
    return clock, opener


def test_llamacpp_health_503_then_200_returns(monkeypatch):
    clock, opener = _prepare(
        monkeypatch,
        [_models_body("llamacpp"), _http_error(503), _FakeResponse()],
    )

    wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert opener.urls.count("http://127.0.0.1:8080/v1/health") == 2
    assert clock.sleeps == [1.0]


def test_models_503_then_non_llama_returns_without_health(monkeypatch):
    _, opener = _prepare(monkeypatch, [_http_error(503), _models_body("vllm")])

    wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert opener.urls == ["http://127.0.0.1:8080/v1/models"] * 2


def test_vllm_models_skip_health(monkeypatch):
    _, opener = _prepare(monkeypatch, [_models_body("vllm")])

    wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert opener.urls == ["http://127.0.0.1:8080/v1/models"]


def test_health_500_exits_with_body(monkeypatch, capsys):
    _prepare(monkeypatch, [_models_body("llamacpp"), _http_error(500, b'{"error": "failed to load model"}')])

    with pytest.raises(SystemExit) as exit_info:
        wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert exit_info.value.code == 1
    assert "failed to load model" in capsys.readouterr().err


def test_wait_deadline_exits(monkeypatch, capsys):
    _prepare(monkeypatch, [urllib.error.URLError("refused")] * 10)

    with pytest.raises(SystemExit) as exit_info:
        wait_for_server_ready(_provider(), connect_budget=3, timeout=180)

    assert exit_info.value.code == 1
    assert "not ready after 3s" in capsys.readouterr().err


def test_models_401_returns_without_retry(monkeypatch):
    clock, opener = _prepare(monkeypatch, [_http_error(401)])

    wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert opener.urls == ["http://127.0.0.1:8080/v1/models"]
    assert clock.sleeps == []


def test_dead_port_exits_after_connect_budget(monkeypatch, capsys):
    clock, _ = _prepare(monkeypatch, [urllib.error.URLError("refused")] * 100)

    with pytest.raises(SystemExit):
        wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert clock.now == 16.0
    assert "not ready after 16s" in capsys.readouterr().err


def test_health_404_falls_back_to_root(monkeypatch):
    _, opener = _prepare(monkeypatch, [_models_body("llamacpp"), _http_error(404), _FakeResponse()])

    wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert opener.urls[-2:] == ["http://127.0.0.1:8080/v1/health", "http://127.0.0.1:8080/health"]


def test_every_probe_passes_a_timeout(monkeypatch):
    _, opener = _prepare(monkeypatch, [_models_body("llamacpp"), _http_error(503), _FakeResponse()])

    wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert all(timeout is not None and 0 < timeout <= 5 for _, timeout in opener.calls)


def _chat_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "dryrun": False,
        "url": "http://127.0.0.1:8080/v1",
        "api_key": None,
        "color": "never",
        "prefix": "> ",
        "list": True,
        "keepalive": None,
        "name": None,
        "server_process": None,
        "engine": None,
        "ARGS": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _patch_chat_boundaries(monkeypatch) -> MagicMock:
    provider = MagicMock()
    provider.list_models.return_value = []
    monkeypatch.setattr(chat_module, "OpenAICompletionsChatProvider", MagicMock(return_value=provider))
    monkeypatch.setattr(chat_module, "ServerMonitor", MagicMock())
    return provider


def test_standalone_chat_waits_for_readiness(monkeypatch):
    provider = _patch_chat_boundaries(monkeypatch)
    wait = MagicMock()
    monkeypatch.setattr(chat_module, "wait_for_server_ready", wait)

    chat_module.chat(_chat_args())

    wait.assert_called_once_with(
        provider,
        connect_budget=chat_module.SERVER_CONNECT_BUDGET,
        timeout=chat_module.SERVER_READY_TIMEOUT,
    )


def test_nocontainer_server_process_gets_full_budget(monkeypatch):
    provider = _patch_chat_boundaries(monkeypatch)
    wait = MagicMock()
    monkeypatch.setattr(chat_module, "wait_for_server_ready", wait)

    chat_module.chat(_chat_args(server_process=MagicMock()))

    wait.assert_called_once_with(
        provider,
        connect_budget=chat_module.SERVER_READY_TIMEOUT,
        timeout=chat_module.SERVER_READY_TIMEOUT,
    )


def test_container_name_skips_readiness_wait(monkeypatch):
    _patch_chat_boundaries(monkeypatch)
    wait = MagicMock()
    monkeypatch.setattr(chat_module, "wait_for_server_ready", wait)

    chat_module.chat(_chat_args(name="ramalama-box", engine="podman"))

    wait.assert_not_called()


def test_explicit_provider_skips_readiness_wait(monkeypatch):
    _patch_chat_boundaries(monkeypatch)
    wait = MagicMock()
    monkeypatch.setattr(chat_module, "wait_for_server_ready", wait)
    provider = MagicMock()
    provider.list_models.return_value = []

    chat_module.chat(_chat_args(), provider=provider)

    wait.assert_not_called()


def test_list_waits_before_listing_models(monkeypatch):
    provider = _patch_chat_boundaries(monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(
        chat_module, "wait_for_server_ready", MagicMock(side_effect=lambda *args, **kwargs: events.append("wait"))
    )
    provider.list_models.side_effect = lambda: events.append("list") or []

    chat_module.chat(_chat_args(list=True))

    assert events == ["wait", "list"]


def test_keepalive_alarm_armed_after_readiness(monkeypatch):
    provider = _patch_chat_boundaries(monkeypatch)
    events: list[object] = []
    monkeypatch.setattr(
        chat_module, "wait_for_server_ready", MagicMock(side_effect=lambda *args, **kwargs: events.append("wait"))
    )
    fake_signal = MagicMock()
    fake_signal.alarm.side_effect = lambda seconds: events.append(("alarm", seconds))
    monkeypatch.setattr(chat_module, "signal", fake_signal)
    provider.list_models.side_effect = lambda: events.append("list") or []

    chat_module.chat(_chat_args(keepalive=1))

    assert events[:2] == ["wait", ("alarm", 1)]


def test_health_phase_deadline_exits_after_detection(monkeypatch, capsys):
    _prepare(monkeypatch, [_models_body("llamacpp")] + [_http_error(503)] * 10)

    with pytest.raises(SystemExit) as exit_info:
        wait_for_server_ready(_provider(), connect_budget=16, timeout=5)

    assert exit_info.value.code == 1
    assert "not ready after 5s" in capsys.readouterr().err


def test_non_http_garbage_on_models_returns(monkeypatch):
    _, opener = _prepare(monkeypatch, [http.client.BadStatusLine("garbage")])

    wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert opener.urls == ["http://127.0.0.1:8080/v1/models"]


def test_not_ready_stops_nocontainer_server_process(monkeypatch):
    _patch_chat_boundaries(monkeypatch)
    monkeypatch.setattr(chat_module, "wait_for_server_ready", MagicMock(side_effect=SystemExit(1)))
    server_process = MagicMock()

    with pytest.raises(SystemExit):
        chat_module.chat(_chat_args(server_process=server_process))

    server_process.terminate.assert_called_once_with()


def test_empty_models_body_retries(monkeypatch):
    _, opener = _prepare(monkeypatch, [_FakeResponse(b""), _models_body("vllm")])

    wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert opener.urls == ["http://127.0.0.1:8080/v1/models"] * 2


def test_remote_disconnect_while_loading_retries(monkeypatch):
    _, opener = _prepare(monkeypatch, [http.client.RemoteDisconnected("closed"), _models_body("vllm")])

    wait_for_server_ready(_provider(), connect_budget=16, timeout=180)

    assert len(opener.urls) == 2
