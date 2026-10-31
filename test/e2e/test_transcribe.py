import re
import socket
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


def _write_silence(path, seconds: float = 1.0) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(16000 * seconds))


@pytest.mark.e2e
@skip_if_no_container
def test_transcribe_dryrun_container_command(tmp_path):
    _require_network()
    audio = tmp_path / "a.wav"
    _write_silence(audio, seconds=0.1)

    result = check_output(["ramalama", "-q", "--runtime", "whisper.cpp", "--dryrun", "transcribe", MODEL, str(audio)])

    assert "--entrypoint=whisper-cli" in result
    assert WHISPER_IMAGE in result
    assert re.search(r"-otxt\s+-of\s+-\s+-np", result)
    assert "--mount=type=bind" in result
    assert "destination=/mnt/audio/input.wav" in result
    assert re.search(r"-f\s+/mnt/audio/input\.wav", result)


@pytest.mark.e2e
def test_transcribe_dryrun_nocontainer_command(tmp_path):
    _require_network()
    audio = tmp_path / "a.wav"
    _write_silence(audio, seconds=0.1)

    result = check_output(
        ["ramalama", "-q", "--runtime", "whisper.cpp", "--dryrun", "--nocontainer", "transcribe", MODEL, str(audio)]
    )

    assert "whisper-cli" in result
    assert "--entrypoint" not in result
    assert str(audio) in result


@pytest.mark.e2e
@skip_if_no_container
def test_transcribe_wav_to_stdout(tmp_path):
    _require_network()

    audio = tmp_path / "silence.wav"
    _write_silence(audio)

    with RamalamaExecWorkspace() as ctx:
        result = ctx.check_output(["ramalama", "-q", "--runtime", "whisper.cpp", "transcribe", MODEL, str(audio)])

    # -np keeps whisper.cpp diagnostics out of the output, leaving the
    # transcript (empty for silence) on stdout.
    assert "whisper_print_timings" not in result
    assert "system_info" not in result
