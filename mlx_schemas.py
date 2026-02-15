"""Base MLX schemas used for requesting chat generation and responses

Models can have different format for messages, tools, system, etc but all contained
in ChatParams, which is the class that is processed for inference.

Parsers (e.g., qwen3_parser) can subclass ChatParams and override its properties as needed.
"""

from typing import TypedDict, Literal, Optional
from pydantic import BaseModel


class ToolFunction(TypedDict):
    name: str
    description: str
    parameters: dict


class ToolUse(TypedDict):
    type: Literal["function"]
    function: ToolFunction


class Message(TypedDict):
    role: str
    content: str


class ChatParams(BaseModel):
    """
    Internal chat representation
    """

    messages: list[Message]
    tools: Optional[list[ToolUse]] = None
    max_tokens: int
    sampler_params: Optional[dict] = None
    enable_thinking: bool = False
    request_model: str
