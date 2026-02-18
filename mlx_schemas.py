"""Base MLX schemas used for requesting chat generation and responses

Models can have different format for messages, tools, system, etc but all contained
in ChatParams, which is the class that is processed for inference.

Parsers (e.g., qwen3_parser) can subclass ChatParams and override its properties as needed.
"""

from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, model_validator


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
    stop_sequences: Optional[list[str]] = None
    enable_thinking: bool = False
    request_model: str
    structured_output_requested: bool = False
    add_generation_prompt: bool = True
    continue_final_message: bool = False

    @model_validator(mode="after")
    def validate_generation_params(self):
        if self.add_generation_prompt and self.continue_final_message:
            raise ValueError(
                "Cannot use both add_generation_prompt and continue_final_message. "
                "add_generation_prompt is for generating new assistant responses, "
                "while continue_final_message is for prefilling/continuing existing responses."
            )
        return self
