"""
deAPI Tools
=============================

Demonstrates deAPI tools for image, video, speech and music generation and audio transcription.

Get an API key at https://app.deapi.ai/register and set it as DEAPI_API_KEY.
"""

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.deapi import DeapiTools

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------


deapi_agent = Agent(
    name="deAPI Media Generator Agent",
    model=OpenAIResponses(id="gpt-5.6-luna"),
    tools=[
        DeapiTools(
            image_model="Flux1schnell",
            video_model="Ltx2_5_22B_Dist_INT8",
        )
    ],
    description="You are an AI agent that can generate images, videos, speech and music, and transcribe audio using the deAPI API.",
    instructions=[
        "When the user asks you to create an image, use the `generate_image` tool.",
        "When the user asks you to change an existing image, use the `edit_image` tool.",
        "When the user asks you to create a video, use the `generate_video` tool.",
        "When the user asks you to read text aloud, use the `generate_speech` tool.",
        "When the user asks you to create music, use the `generate_music` tool.",
        "When the user asks you to transcribe audio or video, use the `transcribe_audio` tool.",
        "Return the URL as raw to the user.",
        "Don't convert the media URL to markdown or anything else.",
    ],
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    deapi_agent.print_response("Generate an image of a lighthouse on a stormy coast")
