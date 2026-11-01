import json
import re
import socket
import subprocess
import urllib.request
import wave

import pytest

from test.conftest import skip_if_no_container
from test.e2e.utils import RamalamaExecWorkspace, check_output

MODEL = "huggingface://ggerganov/whisper.cpp/ggml-base.en.bin"
WHISPER_IMAGE = "ghcr.io/ggml-org/whisper.cpp"


def _network_available() -> bool:
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=3):
            return True
    except OSError:
        return False


def _require_network() -> None:
    if not _network_available():
        pytest.skip("no network access to resolve the whisper.cpp model")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_silence(path, seconds: float = 1.0) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(16000 * seconds))


def _post_audio(port: int, audio) -> dict:
    boundary = "ramalama-e2e-boundary"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{audio.name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + audio.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


@pytest.mark.e2e
@skip_if_no_container
def test_serve_dryrun_container_command():
    _require_network()

    result = check_output(["ramalama", "-q", "--runtime", "whisper.cpp", "--dryrun", "serve", MODEL])

    assert "--entrypoint=whisper-server" in result
    assert WHISPER_IMAGE in result
    assert re.search(r"--host\s+::", result)
    assert re.search(r"--port\s+\d+", result)
    assert re.search(r"-m\s+/path/to/model", result)


@pytest.mark.e2e
def test_serve_dryrun_nocontainer_command():
    _require_network()

    result = check_output(["ramalama", "-q", "--runtime", "whisper.cpp", "--dryrun", "--nocontainer", "serve", MODEL])

    assert "whisper-server" in result
    assert "--entrypoint" not in result
    assert re.search(r"--host\s+127\.0\.0\.1", result)
    assert re.search(r"--port\s+\d+", result)


@pytest.mark.e2e
@skip_if_no_container
def test_detached_serve_answers_inference(tmp_path):
    _require_network()

    audio = tmp_path / "silence.wav"
    _write_silence(audio)
    port = _free_port()
    name = f"whisper-e2e-{port}"

    with RamalamaExecWorkspace() as ctx:
        try:
            # Detached serve returns once /health answers 200.
            ctx.check_output(
                [
                    "ramalama",
                    "-q",
                    "--runtime",
                    "whisper.cpp",
                    "serve",
                    "--detach",
                    "--name",
                    name,
                    "--port",
                    str(port),
                    MODEL,
                ]
            )
            result = _post_audio(port, audio)
        finally:
            try:
                ctx.check_output(["ramalama", "stop", name], stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                pass

    assert "text" in result
