import re
from typing import Any, Dict

from base_chat_parser import BaseChatParser, StreamTextSanitizer
from claude_schemas import (
    # Request models
    ClaudeMessageParams,
    ContentBlock,
    TextBlock,
    TextBlockParam,
    ImageBlockParam,
    ToolResultBlockParam,
)

from mlx_schemas import ChatParams

# Qwen3 recommended sampler defaults
# See: https://huggingface.co/Qwen/Qwen3-14B
QWEN3_THINKING_SAMPLER = {"temp": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
QWEN3_NON_THINKING_SAMPLER = {"temp": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


class Parser(BaseChatParser):
    def parse_chat_params(self, params: ClaudeMessageParams) -> ChatParams:
        enable_thinking = _parse_thinking(params)
        messages = _parse_messages(params)

        add_generation_prompt = True
        continue_final_message = False
        is_prefilling = _has_prefill_message(params)
        structured_output_requested = _has_structured_output_request(params)

        if is_prefilling and not structured_output_requested:
            add_generation_prompt = False
            continue_final_message = True

        return ChatParams(
            messages=messages,
            tools=_parse_tools(params),
            enable_thinking=enable_thinking,
            max_tokens=params.max_tokens,
            sampler_params=_parse_sampler(params, enable_thinking),
            stop_sequences=params.stop_sequences,
            request_model=params.model,
            structured_output_requested=structured_output_requested,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )

    def parse_response_text(self, text: str) -> ContentBlock:
        # TODO: handle other types of content block
        return TextBlock(text=self.sanitize_response_text(text))

    def sanitize_response_text(self, text: str) -> str:
        return _sanitize_qwen3_text(text).strip()

    def create_stream_text_sanitizer(self) -> StreamTextSanitizer:
        return _Qwen3StreamSanitizer()

    def default_stop_sequences(self) -> list[str]:
        return [_IM_END]


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_STREAM_MARKERS = (_THINK_OPEN, _THINK_CLOSE, _IM_START, _IM_END)
_MAX_MARKER_LENGTH = max(len(marker) for marker in _STREAM_MARKERS + (_THINK_CLOSE,))


def _sanitize_qwen3_text(text: str) -> str:
    if not text:
        return ""

    sanitized = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    sanitized = re.sub(r"</?think>", "", sanitized, flags=re.IGNORECASE)
    sanitized = sanitized.replace(_IM_END, "")
    sanitized = re.sub(r"<\|im_start\|>\s*\w*\s*", "", sanitized)
    return sanitized


class _Qwen3StreamSanitizer(StreamTextSanitizer):
    def __init__(self):
        self.buffer = ""
        self.in_think_block = False

    def push(self, chunk: str) -> str:
        if not chunk:
            return ""

        self.buffer += chunk
        output_parts = []

        while True:
            if self.in_think_block:
                close_index = self.buffer.find(_THINK_CLOSE)
                if close_index == -1:
                    self.buffer = self.buffer[-(len(_THINK_CLOSE) - 1) :]
                    break
                self.buffer = self.buffer[close_index + len(_THINK_CLOSE) :]
                self.in_think_block = False
                continue

            marker = self._find_first_marker()
            if marker is None:
                if len(self.buffer) <= _MAX_MARKER_LENGTH - 1:
                    break
                safe_text = self.buffer[: -(_MAX_MARKER_LENGTH - 1)]
                output_parts.append(safe_text)
                self.buffer = self.buffer[-(_MAX_MARKER_LENGTH - 1) :]
                break

            index, token = marker
            if index > 0:
                output_parts.append(self.buffer[:index])
                self.buffer = self.buffer[index:]

            if token == _THINK_OPEN:
                self.buffer = self.buffer[len(_THINK_OPEN) :]
                self.in_think_block = True
                continue

            if token == _THINK_CLOSE:
                self.buffer = self.buffer[len(_THINK_CLOSE) :]
                continue

            if token == _IM_START:
                self.buffer = self.buffer[len(_IM_START) :]
                match = re.match(r"^\s*\w+\s*", self.buffer)
                if match:
                    self.buffer = self.buffer[match.end() :]
                continue

            if token == _IM_END:
                self.buffer = self.buffer[len(_IM_END) :]
                continue

        return "".join(output_parts)

    def finish(self) -> str:
        if self.in_think_block:
            self.buffer = ""
            return ""

        tail = _sanitize_qwen3_text(self.buffer)
        self.buffer = ""
        return tail

    def _find_first_marker(self):
        first_index = None
        first_token = None
        for token in _STREAM_MARKERS:
            index = self.buffer.find(token)
            if index == -1:
                continue
            if first_index is None or index < first_index:
                first_index = index
                first_token = token

        if first_index is None:
            return None

        return first_index, first_token


def _parse_messages(params: ClaudeMessageParams):
    # Put system prompt at the beginning of the chat
    # See https://huggingface.co/docs/transformers/en/chat_templating#using-applychattemplate
    # In the case of qwen3 we have to put all the system messages in messages[0]

    if params.system is not None:
        return [
            _parse_system_prompt(params.system),
            *_parse_conversation(params.messages),
        ]
    return _parse_conversation(params.messages)


def _parse_system_prompt(claude_system_prompt):
    if isinstance(claude_system_prompt, str):
        return {
            "role": "system",
            "content": claude_system_prompt,
        }

    # If not a str, then this should be a list
    text_content = "\n\n".join(
        msg.text for msg in claude_system_prompt if isinstance(msg, TextBlockParam)
    )
    return {"role": "system", "content": text_content}


def _parse_conversation(claude_messages):
    messages = []
    for msg in claude_messages:
        content_text = []
        role = msg.role

        for c in msg.content:
            if isinstance(c, str):
                content_text.append(c)

            elif isinstance(c, TextBlockParam):
                content_text.append(c.text)

            elif isinstance(c, ImageBlockParam):
                content_text.append(
                    f"[Image Context: {c.source:50}] ... (Omitted: Text-only mode)"
                )

            elif isinstance(c, ToolResultBlockParam):
                content_text.append(
                    f"<tool_response>{_parse_tool_result(c.content)}</tool_response>"
                )

            else:
                raise TypeError("Can't parse unknown content type in Claude message")

        messages.append({"role": role, "content": "\n\n".join(content_text)})

    return messages


def _parse_tool_result(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, TextBlockParam):
        return content.text
    elif isinstance(content, ImageBlockParam):
        return content.source


def _parse_tools(params: ClaudeMessageParams):
    if not params.tools:
        return None
    tools = []
    for tool in params.tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
        )
    return tools


def _parse_thinking(params: ClaudeMessageParams):
    if not params.thinking or params.thinking.get("type") == "disabled":
        return False

    if params.thinking:
        return True


def _parse_sampler(params: ClaudeMessageParams, enable_thinking: bool):
    defaults = QWEN3_THINKING_SAMPLER if enable_thinking else QWEN3_NON_THINKING_SAMPLER

    sampler_params = {**defaults}

    # NOTE: We intentionally do NOT override temperature from the request.
    # Claude Code always sends a temperature value (default 1.0) which would
    # break Qwen3's recommended defaults (0.6 thinking / 0.7 non-thinking).

    if params.top_k is not None:
        sampler_params["top_k"] = params.top_k

    if params.top_p is not None:
        sampler_params["top_p"] = params.top_p

    return sampler_params


def _has_prefill_message(params: ClaudeMessageParams) -> bool:
    if len(params.messages) == 0:
        return False

    last_message = params.messages[-1]
    if last_message.role != "assistant":
        return False

    if isinstance(last_message.content, str):
        return bool(last_message.content.strip())

    return len(last_message.content) > 0


def _has_structured_output_request(params: ClaudeMessageParams) -> bool:
    if params.response_format:
        response_format_type = params.response_format.get("type")
        if response_format_type in {"json_schema", "json_object"}:
            return True

    if params.tool_choice and params.tools:
        tool_choice_type = params.tool_choice.get("type")
        if tool_choice_type in {"tool", "any", "required"}:
            return True

    return False
