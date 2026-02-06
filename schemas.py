from typing import Dict, Any, Optional, Union, Literal
from pydantic import BaseModel


# Request models


class TextBlockParam(BaseModel):
    type: Literal["text"]
    text: str


class ImageBlockParam(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ToolUseBlockParam(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultBlockParam(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, TextBlockParam, ImageBlockParam]


class SystemContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingConfig(BaseModel):
    budget_tokens: Optional[int]
    type: Literal["enabled", "disabled"] = "disabled"


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


ContentBlockParam = Union[
    TextBlockParam,
    ImageBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, list[ContentBlockParam]]


class ClaudeMessageParams(BaseModel):
    max_tokens: int
    messages: list[Message]
    model: str
    metadata: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[list[str]] = None
    stream: Optional[bool] = False
    system: Optional[Union[str, list[SystemContent]]] = None
    temperature: Optional[float] = 1.0
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    tools: Optional[list[Tool]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    original_model: Optional[str] = None


class ClaudeTokenCountParams(BaseModel):
    model: str
    messages: list[Message]
    system: Optional[Union[str, list[SystemContent]]] = None
    tools: Optional[list[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


# Response models
class ClaudeMessage(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    # content: list[ContentBlockText]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: Optional[str] = None
    usage: Usage


class ClaudeTokenCount(BaseModel):
    input_tokens: int


# class MessageStreamResponse(BaseModel):
#     type: str
#     index: Optional[int] = None
#     delta: Optional[Dict[str, Any]] = None
#     usage: Optional[Usage] = None
