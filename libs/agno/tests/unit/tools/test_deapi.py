"""Unit tests for DeapiTools (agno.tools.deapi)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agno.tools.deapi import DEAPI_BASE_URL, DeapiTools
from agno.tools.function import ToolResult


def _response(data, method="GET", status_code=200, url="https://api.deapi.ai"):
    request = httpx.Request(method, url)
    return httpx.Response(status_code, json=data, request=request)


def _submitted(request_id="req-123"):
    return _response({"data": {"request_id": request_id}}, method="POST")


def _job(status, **fields):
    return _response({"data": {"status": status, **fields}})


@pytest.fixture
def tools():
    return DeapiTools(api_key="test-key", poll_interval=0, max_wait_time=10)


def test_init_registers_tools():
    toolkit = DeapiTools(api_key="test-key")
    expected = {
        "generate_image",
        "edit_image",
        "generate_video",
        "generate_speech",
        "generate_music",
        "transcribe_audio",
    }
    assert expected == set(toolkit.functions)
    assert expected == set(toolkit.async_functions)


def test_init_enable_flags():
    toolkit = DeapiTools(
        api_key="test-key",
        enable_edit_image=False,
        enable_generate_video=False,
        enable_generate_speech=False,
        enable_generate_music=False,
        enable_transcribe_audio=False,
    )
    assert list(toolkit.functions) == ["generate_image"]


def test_init_all_overrides_flags():
    toolkit = DeapiTools(api_key="test-key", enable_generate_image=False, enable_generate_video=False, all=True)
    assert "generate_image" in toolkit.functions
    assert "generate_video" in toolkit.functions


def test_init_reads_api_key_from_env():
    with patch.dict("os.environ", {"DEAPI_API_KEY": "env-key"}):
        toolkit = DeapiTools()
    assert toolkit.api_key == "env-key"
    assert toolkit._headers["Authorization"] == "Bearer env-key"


def test_missing_api_key_returns_error():
    with patch.dict("os.environ", {}, clear=True):
        toolkit = DeapiTools()
    with patch("agno.tools.deapi.httpx.post") as mock_post:
        result = toolkit.generate_image("a red apple")
    assert result.content == "Error: DEAPI_API_KEY not set"
    mock_post.assert_not_called()


def test_generate_image_success(tools):
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()) as mock_post,
        patch(
            "agno.tools.deapi.httpx.get",
            side_effect=[_job("processing"), _job("done", result_url="https://cdn.example.com/image.png")],
        ) as mock_get,
        patch("agno.tools.deapi.time.sleep") as mock_sleep,
    ):
        result = tools.generate_image("a red apple")

    assert isinstance(result, ToolResult)
    assert "https://cdn.example.com/image.png" in result.content
    assert result.images is not None
    assert len(result.images) == 1
    assert result.images[0].url == "https://cdn.example.com/image.png"
    assert result.images[0].original_prompt == "a red apple"

    assert mock_post.call_args.args[0] == f"{DEAPI_BASE_URL}/images/generations"
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert mock_post.call_args.kwargs["json"] == {
        "prompt": "a red apple",
        "model": "Flux1schnell",
        "width": 768,
        "height": 768,
        "steps": 4,
        "seed": -1,
    }
    assert mock_get.call_count == 2
    assert mock_get.call_args.args[0] == f"{DEAPI_BASE_URL}/jobs/req-123"
    mock_sleep.assert_called_once_with(0)


def test_generate_image_custom_model_and_size():
    toolkit = DeapiTools(api_key="test-key", image_model="ZImageTurbo_INT8", image_steps=8, poll_interval=0)
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()) as mock_post,
        patch("agno.tools.deapi.httpx.get", return_value=_job("done", result_url="https://cdn.example.com/i.png")),
    ):
        toolkit.generate_image("a red apple", width=1024, height=512)

    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "ZImageTurbo_INT8"
    assert body["steps"] == 8
    assert (body["width"], body["height"]) == (1024, 512)


def test_generate_image_job_error(tools):
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()),
        patch("agno.tools.deapi.httpx.get", return_value=_job("error", error_reason="NSFW content detected")),
    ):
        result = tools.generate_image("a red apple")

    assert result.content.startswith("Error:")
    assert "NSFW content detected" in result.content
    assert result.images is None


def test_generate_image_validation_error(tools):
    error = _response(
        {"message": "The steps field is required.", "errors": {"steps": ["The steps field is required."]}},
        method="POST",
        status_code=422,
    )
    with patch("agno.tools.deapi.httpx.post", return_value=error):
        result = tools.generate_image("a red apple")

    assert result.content == "Error: HTTP 422: The steps field is required."


def test_generate_image_missing_request_id(tools):
    with patch("agno.tools.deapi.httpx.post", return_value=_response({"data": {}}, method="POST")):
        result = tools.generate_image("a red apple")

    assert result.content == "Error: No request ID returned"


def test_generate_image_empty_output(tools):
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()),
        patch("agno.tools.deapi.httpx.get", return_value=_job("done", result_url=None)),
    ):
        result = tools.generate_image("a red apple")

    assert result.content == "No output received from the model."
    assert result.images is None


def test_polling_retries_transient_errors(tools):
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()),
        patch(
            "agno.tools.deapi.httpx.get",
            side_effect=[
                _response({"message": "Too Many Attempts."}, status_code=429),
                _response({"message": "Server Error"}, status_code=502),
                _job("done", result_url="https://cdn.example.com/image.png"),
            ],
        ) as mock_get,
        patch("agno.tools.deapi.time.sleep"),
    ):
        result = tools.generate_image("a red apple")

    assert result.images is not None
    assert mock_get.call_count == 3


def test_polling_never_waits_past_max_wait_time():
    toolkit = DeapiTools(api_key="test-key", poll_interval=5, max_wait_time=10, request_timeout=30)
    clock = {"now": 100.0}
    sleeps = []

    def fake_get(*args, **kwargs):
        clock["now"] += 8.0
        return _job("processing")

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()),
        patch("agno.tools.deapi.httpx.get", side_effect=fake_get) as mock_get,
        patch("agno.tools.deapi.time.monotonic", side_effect=lambda: clock["now"]),
        patch("agno.tools.deapi.time.sleep", side_effect=fake_sleep),
    ):
        result = toolkit.generate_video("a very slow render")

    assert result.content == "Error: Job req-123 did not finish within 10 seconds"
    # The poll request timeout is capped by the remaining budget.
    assert mock_get.call_args.kwargs["timeout"] == 10
    # The sleep is clamped to the time left rather than the full poll interval.
    assert sleeps == [2.0]
    assert mock_get.call_count == 1


def test_edit_image_uploads_source_image(tools):
    done = _job("done", result_url="https://cdn.example.com/edited.png")
    with (
        patch(
            "agno.tools.deapi.fetch_public_page", return_value=("https://example.com/fox.png", b"png-bytes")
        ) as mock_fetch,
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()) as mock_post,
        patch("agno.tools.deapi.httpx.get", return_value=done),
    ):
        result = tools.edit_image("make the fox blue", image_url="https://example.com/fox.png")

    assert result.images is not None
    assert result.images[0].url == "https://cdn.example.com/edited.png"
    assert "edited successfully" in result.content
    assert mock_fetch.call_args.args[0] == "https://example.com/fox.png"
    assert mock_post.call_args.args[0] == f"{DEAPI_BASE_URL}/images/edits"
    assert mock_post.call_args.kwargs["data"] == {
        "prompt": "make the fox blue",
        "model": "Flux_2_Klein_4B_BF16",
        "steps": 4,
        "seed": -1,
    }
    assert mock_post.call_args.kwargs["files"] == {"image": ("image", b"png-bytes", "application/octet-stream")}


def test_edit_image_source_download_fails(tools):
    with (
        patch("agno.tools.deapi.fetch_public_page", side_effect=httpx.HTTPError("404 Not Found")),
        patch("agno.tools.deapi.httpx.post") as mock_post,
    ):
        result = tools.edit_image("make the fox blue", image_url="https://example.com/missing.png")

    assert result.content.startswith("Error:")
    mock_post.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a.png",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost/a.png",
        "file:///etc/passwd",
    ],
)
def test_edit_image_rejects_non_public_urls(tools, url):
    with patch("agno.tools.deapi.httpx.post") as mock_post:
        result = tools.edit_image("make the fox blue", image_url=url)

    assert result.content.startswith("Error:")
    mock_post.assert_not_called()


def test_generate_video_success(tools):
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()) as mock_post,
        patch("agno.tools.deapi.httpx.get", return_value=_job("done", result_url="https://cdn.example.com/v.mp4")),
    ):
        result = tools.generate_video("a fox running in snow")

    assert result.videos is not None
    assert result.videos[0].url == "https://cdn.example.com/v.mp4"
    assert result.videos[0].mime_type == "video/mp4"
    assert mock_post.call_args.args[0] == f"{DEAPI_BASE_URL}/videos/generations"
    assert mock_post.call_args.kwargs["json"] == {
        "prompt": "a fox running in snow",
        "model": "Ltx2_5_22B_Dist_INT8",
        "width": 768,
        "height": 768,
        "frames": 120,
        "fps": 24,
        "seed": -1,
    }


def test_generate_speech_success(tools):
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()) as mock_post,
        patch("agno.tools.deapi.httpx.get", return_value=_job("done", result_url="https://cdn.example.com/s.mp3")),
    ):
        result = tools.generate_speech("Hello world", voice="af_bella")

    assert result.audios is not None
    assert result.audios[0].url == "https://cdn.example.com/s.mp3"
    assert result.audios[0].mime_type == "audio/mpeg"
    assert mock_post.call_args.args[0] == f"{DEAPI_BASE_URL}/audio/speech"
    assert mock_post.call_args.kwargs["data"] == {
        "text": "Hello world",
        "model": "Kokoro",
        "voice": "af_bella",
        "lang": "en-us",
        "speed": 1.0,
        "format": "mp3",
        "sample_rate": 24000,
    }


def test_generate_speech_default_voice(tools):
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()) as mock_post,
        patch("agno.tools.deapi.httpx.get", return_value=_job("done", result_url="https://cdn.example.com/s.mp3")),
    ):
        tools.generate_speech("Hello world")

    assert mock_post.call_args.kwargs["data"]["voice"] == "af_alloy"


def test_generate_music_success(tools):
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()) as mock_post,
        patch("agno.tools.deapi.httpx.get", return_value=_job("done", result_url="https://cdn.example.com/m.mp3")),
    ):
        result = tools.generate_music("calm lo-fi piano", duration=10)

    assert result.audios is not None
    assert result.audios[0].url == "https://cdn.example.com/m.mp3"
    assert mock_post.call_args.args[0] == f"{DEAPI_BASE_URL}/audio/music"
    assert mock_post.call_args.kwargs["data"] == {
        "caption": "calm lo-fi piano",
        "lyrics": "[Instrumental]",
        "model": "AceStep_1_5_XL_Turbo_INT8",
        "duration": 10,
        "inference_steps": 8,
        "guidance_scale": 1,
        "format": "mp3",
        "seed": -1,
    }


def test_transcribe_audio_success(tools):
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()) as mock_post,
        patch("agno.tools.deapi.httpx.get", return_value=_job("done", result="Hello from deAPI.")),
    ):
        result = tools.transcribe_audio("https://www.youtube.com/watch?v=jNQXAC9IVRw")

    assert result.content == "Hello from deAPI."
    assert mock_post.call_args.args[0] == f"{DEAPI_BASE_URL}/audio/transcriptions"
    assert mock_post.call_args.kwargs["data"] == {
        "source_url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        "model": "WhisperLargeV3",
        "include_ts": "false",
        "return_result_in_response": "true",
    }


def test_transcribe_audio_empty_result(tools):
    with (
        patch("agno.tools.deapi.httpx.post", return_value=_submitted()),
        patch("agno.tools.deapi.httpx.get", return_value=_job("done", result=None)),
    ):
        result = tools.transcribe_audio("https://example.com/a.mp3")

    assert result.content == "No transcript received from the model."


def _async_client_factory(client):
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = client
    factory.return_value.__aexit__.return_value = False
    return factory


async def test_agenerate_image_success(tools):
    client = MagicMock()
    client.post = AsyncMock(return_value=_submitted())
    client.get = AsyncMock(side_effect=[_job("pending"), _job("done", result_url="https://cdn.example.com/image.png")])

    with (
        patch("agno.tools.deapi.httpx.AsyncClient", _async_client_factory(client)),
        patch("agno.tools.deapi.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        result = await tools.agenerate_image("a red apple")

    assert result.images is not None
    assert result.images[0].url == "https://cdn.example.com/image.png"
    assert client.post.call_args.args[0] == f"{DEAPI_BASE_URL}/images/generations"
    assert client.get.call_args.args[0] == f"{DEAPI_BASE_URL}/jobs/req-123"
    mock_sleep.assert_awaited_once_with(0)


async def test_atranscribe_audio_job_error(tools):
    client = MagicMock()
    client.post = AsyncMock(return_value=_submitted())
    client.get = AsyncMock(return_value=_job("error", error_code="INVALID_SOURCE"))

    with patch("agno.tools.deapi.httpx.AsyncClient", _async_client_factory(client)):
        result = await tools.atranscribe_audio("https://example.com/a.mp3")

    assert result.content == "Error: Job req-123 failed: INVALID_SOURCE"


async def test_aedit_image_uploads_source_image(tools):
    client = MagicMock()
    client.post = AsyncMock(return_value=_submitted())
    client.get = AsyncMock(return_value=_job("done", result_url="https://cdn.example.com/edited.png"))
    fetch = AsyncMock(return_value=("https://example.com/fox.png", b"png-bytes"))

    with (
        patch("agno.tools.deapi.afetch_public_page", fetch),
        patch("agno.tools.deapi.httpx.AsyncClient", _async_client_factory(client)),
    ):
        result = await tools.aedit_image("make the fox blue", image_url="https://example.com/fox.png")

    assert result.images is not None
    assert result.images[0].url == "https://cdn.example.com/edited.png"
    assert client.post.call_args.kwargs["files"] == {"image": ("image", b"png-bytes", "application/octet-stream")}
