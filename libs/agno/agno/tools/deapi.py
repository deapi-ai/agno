import asyncio
import time
from os import getenv
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import httpx

from agno.knowledge.reader.utils.public_fetch import afetch_public_page, fetch_public_page
from agno.media import Audio, Image, Video
from agno.tools import Toolkit
from agno.tools.function import ToolResult
from agno.utils.log import log_debug, log_error, logger

DEAPI_BASE_URL = "https://api.deapi.ai/api/v2"

# Terminal statuses returned by GET /jobs/{request_id}
DEAPI_DONE_STATUS = "done"
DEAPI_ERROR_STATUS = "error"
MAX_SOURCE_IMAGE_BYTES = 20 * 1024 * 1024

# A deAPI job request: (path, json body, form data, files)
JobRequest = Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]


class DeapiTools(Toolkit):
    """Tools for generating images, videos, speech and music, and transcribing audio with the deAPI API.

    Get an API key at https://app.deapi.ai/register and set it as the DEAPI_API_KEY environment variable.
    See https://docs.deapi.ai for the model catalog.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEAPI_BASE_URL,
        image_model: str = "Flux1schnell",
        image_steps: int = 4,
        edit_model: str = "Flux_2_Klein_4B_BF16",
        edit_steps: int = 4,
        video_model: str = "Ltx2_5_22B_Dist_INT8",
        video_width: int = 768,
        video_height: int = 768,
        video_frames: int = 120,
        video_fps: int = 24,
        speech_model: str = "Kokoro",
        speech_voice: str = "af_alloy",
        speech_lang: str = "en-us",
        music_model: str = "AceStep_1_5_XL_Turbo_INT8",
        transcription_model: str = "WhisperLargeV3",
        poll_interval: float = 2.0,
        max_wait_time: float = 600.0,
        request_timeout: float = 60.0,
        enable_generate_image: bool = True,
        enable_edit_image: bool = True,
        enable_generate_video: bool = True,
        enable_generate_speech: bool = True,
        enable_generate_music: bool = True,
        enable_transcribe_audio: bool = True,
        all: bool = False,
        **kwargs,
    ):
        self.api_key = api_key or getenv("DEAPI_API_KEY")
        if not self.api_key:
            log_error("DEAPI_API_KEY not set. Please set the DEAPI_API_KEY environment variable.")

        self.base_url = base_url.rstrip("/")
        self.image_model = image_model
        self.image_steps = image_steps
        self.edit_model = edit_model
        self.edit_steps = edit_steps
        self.video_model = video_model
        self.video_width = video_width
        self.video_height = video_height
        self.video_frames = video_frames
        self.video_fps = video_fps
        self.speech_model = speech_model
        self.speech_voice = speech_voice
        self.speech_lang = speech_lang
        self.music_model = music_model
        self.transcription_model = transcription_model
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time
        self.request_timeout = request_timeout

        tools: List[Any] = []
        async_tools: List[Tuple[Any, str]] = []
        if all or enable_generate_image:
            tools.append(self.generate_image)
            async_tools.append((self.agenerate_image, "generate_image"))
        if all or enable_edit_image:
            tools.append(self.edit_image)
            async_tools.append((self.aedit_image, "edit_image"))
        if all or enable_generate_video:
            tools.append(self.generate_video)
            async_tools.append((self.agenerate_video, "generate_video"))
        if all or enable_generate_speech:
            tools.append(self.generate_speech)
            async_tools.append((self.agenerate_speech, "generate_speech"))
        if all or enable_generate_music:
            tools.append(self.generate_music)
            async_tools.append((self.agenerate_music, "generate_music"))
        if all or enable_transcribe_audio:
            tools.append(self.transcribe_audio)
            async_tools.append((self.atranscribe_audio, "transcribe_audio"))

        super().__init__(name="deapi_tools", tools=tools, async_tools=async_tools, **kwargs)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "agno-deapi-tools",
        }

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            message = response.json().get("message")
        except ValueError:
            message = None
        return f"HTTP {response.status_code}: {message or response.text[:300]}"

    def _parse_submit_response(self, response: httpx.Response) -> str:
        if response.is_error:
            raise RuntimeError(self._error_message(response))
        request_id = (response.json().get("data") or {}).get("request_id")
        if not request_id:
            raise RuntimeError("No request ID returned")
        return request_id

    def _parse_job_response(self, request_id: str, response: httpx.Response) -> Optional[Dict[str, Any]]:
        """Return the job data once it is done, None while it is still running. Raise if it failed."""
        if response.status_code == 429 or response.status_code >= 500:
            # Transient errors while polling: keep waiting instead of failing the job.
            log_debug(f"deAPI job {request_id} poll returned {response.status_code}, retrying")
            return None
        if response.is_error:
            raise RuntimeError(self._error_message(response))
        job = response.json().get("data") or {}
        status = job.get("status")
        if status == DEAPI_DONE_STATUS:
            return job
        if status == DEAPI_ERROR_STATUS:
            reason = job.get("error_reason") or job.get("error_code") or "unknown error"
            raise RuntimeError(f"Job {request_id} failed: {reason}")
        log_debug(f"deAPI job {request_id} status: {status}")
        return None

    def _timeout_error(self, request_id: str) -> TimeoutError:
        return TimeoutError(f"Job {request_id} did not finish within {self.max_wait_time} seconds")

    def _run_job(self, request: JobRequest) -> Dict[str, Any]:
        """Submit a job to deAPI and poll GET /jobs/{request_id} until it finishes or max_wait_time elapses."""
        if not self.api_key:
            raise ValueError("DEAPI_API_KEY not set")
        path, json, data, files = request
        response = httpx.post(
            f"{self.base_url}{path}",
            headers=self._headers,
            json=json,
            data=data,
            files=files,
            timeout=self.request_timeout,
        )
        request_id = self._parse_submit_response(response)

        deadline = time.monotonic() + self.max_wait_time
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timeout_error(request_id)
            job_response = httpx.get(
                f"{self.base_url}/jobs/{request_id}",
                headers=self._headers,
                timeout=min(self.request_timeout, remaining),
            )
            job = self._parse_job_response(request_id, job_response)
            if job is not None:
                return job
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timeout_error(request_id)
            time.sleep(min(self.poll_interval, remaining))

    async def _arun_job(self, request: JobRequest) -> Dict[str, Any]:
        """Async variant of _run_job."""
        if not self.api_key:
            raise ValueError("DEAPI_API_KEY not set")
        path, json, data, files = request
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(
                f"{self.base_url}{path}", headers=self._headers, json=json, data=data, files=files
            )
            request_id = self._parse_submit_response(response)

            deadline = time.monotonic() + self.max_wait_time
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._timeout_error(request_id)
                job_response = await client.get(
                    f"{self.base_url}/jobs/{request_id}",
                    headers=self._headers,
                    timeout=min(self.request_timeout, remaining),
                )
                job = self._parse_job_response(request_id, job_response)
                if job is not None:
                    return job
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._timeout_error(request_id)
                await asyncio.sleep(min(self.poll_interval, remaining))

    # ------------------------------------------------------------------
    # Request builders and result parsers shared by sync and async tools
    # ------------------------------------------------------------------

    def _image_request(self, prompt: str, width: int, height: int) -> JobRequest:
        body = {
            "prompt": prompt,
            "model": self.image_model,
            "width": width,
            "height": height,
            "steps": self.image_steps,
            "seed": -1,
        }
        return "/images/generations", body, None, None

    def _edit_request(self, prompt: str, image: bytes) -> JobRequest:
        data = {"prompt": prompt, "model": self.edit_model, "steps": self.edit_steps, "seed": -1}
        files = {"image": ("image", image, "application/octet-stream")}
        return "/images/edits", None, data, files

    def _video_request(self, prompt: str) -> JobRequest:
        body = {
            "prompt": prompt,
            "model": self.video_model,
            "width": self.video_width,
            "height": self.video_height,
            "frames": self.video_frames,
            "fps": self.video_fps,
            "seed": -1,
        }
        return "/videos/generations", body, None, None

    def _speech_request(self, text: str, voice: Optional[str], speed: float) -> JobRequest:
        data = {
            "text": text,
            "model": self.speech_model,
            "voice": voice or self.speech_voice,
            "lang": self.speech_lang,
            "speed": speed,
            "format": "mp3",
            "sample_rate": 24000,
        }
        return "/audio/speech", None, data, None

    def _music_request(self, caption: str, lyrics: str, duration: int) -> JobRequest:
        data = {
            "caption": caption,
            "lyrics": lyrics,
            "model": self.music_model,
            "duration": duration,
            "inference_steps": 8,
            "guidance_scale": 1,
            "format": "mp3",
            "seed": -1,
        }
        return "/audio/music", None, data, None

    def _transcription_request(self, audio_url: str) -> JobRequest:
        data = {
            "source_url": audio_url,
            "model": self.transcription_model,
            "include_ts": "false",
            "return_result_in_response": "true",
        }
        return "/audio/transcriptions", None, data, None

    @staticmethod
    def _image_result(job: Dict[str, Any], prompt: str, action: str) -> ToolResult:
        url = job.get("result_url")
        if not url:
            return ToolResult(content="No output received from the model.")
        image = Image(id=str(uuid4()), url=url, original_prompt=prompt)
        return ToolResult(content=f"Image {action} successfully: {url}", images=[image])

    @staticmethod
    def _video_result(job: Dict[str, Any], prompt: str) -> ToolResult:
        url = job.get("result_url")
        if not url:
            return ToolResult(content="No output received from the model.")
        video = Video(id=str(uuid4()), url=url, original_prompt=prompt, mime_type="video/mp4")
        return ToolResult(content=f"Video generated successfully: {url}", videos=[video])

    @staticmethod
    def _audio_result(job: Dict[str, Any], label: str) -> ToolResult:
        url = job.get("result_url")
        if not url:
            return ToolResult(content="No output received from the model.")
        audio = Audio(id=str(uuid4()), url=url, mime_type="audio/mpeg")
        return ToolResult(content=f"{label} generated successfully: {url}", audios=[audio])

    @staticmethod
    def _transcription_result(job: Dict[str, Any]) -> ToolResult:
        transcript = job.get("result")
        if not transcript:
            return ToolResult(content="No transcript received from the model.")
        return ToolResult(content=transcript)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def generate_image(self, prompt: str, width: int = 768, height: int = 768) -> ToolResult:
        """
        Use this function to generate an image from a text prompt using deAPI.

        Args:
            prompt (str): A text description of the image to generate.
            width (int): Image width in pixels. Defaults to 768.
            height (int): Image height in pixels. Defaults to 768.

        Returns:
            ToolResult: Contains the generated image and success message.
        """
        try:
            job = self._run_job(self._image_request(prompt, width, height))
            return self._image_result(job, prompt, "generated")
        except Exception as e:
            logger.exception("Failed to generate image")
            return ToolResult(content=f"Error: {e}")

    async def agenerate_image(self, prompt: str, width: int = 768, height: int = 768) -> ToolResult:
        """
        Use this function to generate an image from a text prompt using deAPI.

        Args:
            prompt (str): A text description of the image to generate.
            width (int): Image width in pixels. Defaults to 768.
            height (int): Image height in pixels. Defaults to 768.

        Returns:
            ToolResult: Contains the generated image and success message.
        """
        try:
            job = await self._arun_job(self._image_request(prompt, width, height))
            return self._image_result(job, prompt, "generated")
        except Exception as e:
            logger.exception("Failed to generate image")
            return ToolResult(content=f"Error: {e}")

    def edit_image(self, prompt: str, image_url: str) -> ToolResult:
        """
        Use this function to edit an existing image with a text instruction using deAPI.

        Args:
            prompt (str): A text description of the change to make to the image.
            image_url (str): Public URL of the image to edit.

        Returns:
            ToolResult: Contains the edited image and success message.
        """
        try:
            # Only public addresses are fetched: the URL may come from model output.
            _, image = fetch_public_page(image_url, max_bytes=MAX_SOURCE_IMAGE_BYTES)
            job = self._run_job(self._edit_request(prompt, image))
            return self._image_result(job, prompt, "edited")
        except Exception as e:
            logger.exception("Failed to edit image")
            return ToolResult(content=f"Error: {e}")

    async def aedit_image(self, prompt: str, image_url: str) -> ToolResult:
        """
        Use this function to edit an existing image with a text instruction using deAPI.

        Args:
            prompt (str): A text description of the change to make to the image.
            image_url (str): Public URL of the image to edit.

        Returns:
            ToolResult: Contains the edited image and success message.
        """
        try:
            # Only public addresses are fetched: the URL may come from model output.
            _, image = await afetch_public_page(image_url, max_bytes=MAX_SOURCE_IMAGE_BYTES)
            job = await self._arun_job(self._edit_request(prompt, image))
            return self._image_result(job, prompt, "edited")
        except Exception as e:
            logger.exception("Failed to edit image")
            return ToolResult(content=f"Error: {e}")

    def generate_video(self, prompt: str) -> ToolResult:
        """
        Use this function to generate a short video from a text prompt using deAPI.

        Args:
            prompt (str): A text description of the video to generate.

        Returns:
            ToolResult: Contains the generated video and success message.
        """
        try:
            job = self._run_job(self._video_request(prompt))
            return self._video_result(job, prompt)
        except Exception as e:
            logger.exception("Failed to generate video")
            return ToolResult(content=f"Error: {e}")

    async def agenerate_video(self, prompt: str) -> ToolResult:
        """
        Use this function to generate a short video from a text prompt using deAPI.

        Args:
            prompt (str): A text description of the video to generate.

        Returns:
            ToolResult: Contains the generated video and success message.
        """
        try:
            job = await self._arun_job(self._video_request(prompt))
            return self._video_result(job, prompt)
        except Exception as e:
            logger.exception("Failed to generate video")
            return ToolResult(content=f"Error: {e}")

    def generate_speech(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> ToolResult:
        """
        Use this function to convert text to speech using deAPI.

        Args:
            text (str): The text to speak.
            voice (str): Optional voice id to use instead of the default voice (e.g. af_alloy, af_bella, am_adam).
            speed (float): Speech speed, from 0.5 to 2.0. Defaults to 1.0.

        Returns:
            ToolResult: Contains the generated audio and success message.
        """
        try:
            job = self._run_job(self._speech_request(text, voice, speed))
            return self._audio_result(job, "Speech")
        except Exception as e:
            logger.exception("Failed to generate speech")
            return ToolResult(content=f"Error: {e}")

    async def agenerate_speech(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> ToolResult:
        """
        Use this function to convert text to speech using deAPI.

        Args:
            text (str): The text to speak.
            voice (str): Optional voice id to use instead of the default voice (e.g. af_alloy, af_bella, am_adam).
            speed (float): Speech speed, from 0.5 to 2.0. Defaults to 1.0.

        Returns:
            ToolResult: Contains the generated audio and success message.
        """
        try:
            job = await self._arun_job(self._speech_request(text, voice, speed))
            return self._audio_result(job, "Speech")
        except Exception as e:
            logger.exception("Failed to generate speech")
            return ToolResult(content=f"Error: {e}")

    def generate_music(self, caption: str, lyrics: str = "[Instrumental]", duration: int = 30) -> ToolResult:
        """
        Use this function to generate a music track from a text description using deAPI.

        Args:
            caption (str): A description of the music style, mood and instruments.
            lyrics (str): Song lyrics, or "[Instrumental]" for a track without vocals.
            duration (int): Track length in seconds, from 10 to 300. Defaults to 30.

        Returns:
            ToolResult: Contains the generated audio and success message.
        """
        try:
            job = self._run_job(self._music_request(caption, lyrics, duration))
            return self._audio_result(job, "Music")
        except Exception as e:
            logger.exception("Failed to generate music")
            return ToolResult(content=f"Error: {e}")

    async def agenerate_music(self, caption: str, lyrics: str = "[Instrumental]", duration: int = 30) -> ToolResult:
        """
        Use this function to generate a music track from a text description using deAPI.

        Args:
            caption (str): A description of the music style, mood and instruments.
            lyrics (str): Song lyrics, or "[Instrumental]" for a track without vocals.
            duration (int): Track length in seconds, from 10 to 300. Defaults to 30.

        Returns:
            ToolResult: Contains the generated audio and success message.
        """
        try:
            job = await self._arun_job(self._music_request(caption, lyrics, duration))
            return self._audio_result(job, "Music")
        except Exception as e:
            logger.exception("Failed to generate music")
            return ToolResult(content=f"Error: {e}")

    def transcribe_audio(self, audio_url: str) -> ToolResult:
        """
        Use this function to transcribe speech from an audio or video URL (including YouTube links) using deAPI.

        Args:
            audio_url (str): Public URL of the audio or video to transcribe.

        Returns:
            ToolResult: Contains the transcript text.
        """
        try:
            job = self._run_job(self._transcription_request(audio_url))
            return self._transcription_result(job)
        except Exception as e:
            logger.exception("Failed to transcribe audio")
            return ToolResult(content=f"Error: {e}")

    async def atranscribe_audio(self, audio_url: str) -> ToolResult:
        """
        Use this function to transcribe speech from an audio or video URL (including YouTube links) using deAPI.

        Args:
            audio_url (str): Public URL of the audio or video to transcribe.

        Returns:
            ToolResult: Contains the transcript text.
        """
        try:
            job = await self._arun_job(self._transcription_request(audio_url))
            return self._transcription_result(job)
        except Exception as e:
            logger.exception("Failed to transcribe audio")
            return ToolResult(content=f"Error: {e}")
