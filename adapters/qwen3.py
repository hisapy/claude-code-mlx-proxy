import re
import json

from base_chat_parser import BaseChatParser, StreamSegmentParser
from claude_schemas import (
    # Request models
    ClaudeMessageParams,
    TextBlockParam,
    ImageBlockParam,
    ToolUseBlockParam,
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

    def sanitize_response_text(self, text: str) -> str:
        return _sanitize_qwen3_text(text).strip()

    def create_stream_segment_parser(
        self, enable_thinking: bool = False
    ) -> StreamSegmentParser:
        return _Qwen3StreamSegmentParser(enable_thinking=enable_thinking)

    def default_stop_sequences(self) -> list[str]:
        return [_IM_END]


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def _sanitize_qwen3_text(text: str) -> str:
    if not text:
        return ""

    sanitized = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    sanitized = re.sub(r"</?think>", "", sanitized, flags=re.IGNORECASE)
    sanitized = sanitized.replace(_IM_END, "")
    sanitized = re.sub(r"<\|im_start\|>\s*\w*\s*", "", sanitized)
    return sanitized


class _Qwen3StreamSegmentParser(StreamSegmentParser):
    def __init__(self, enable_thinking: bool):
        self.enable_thinking = enable_thinking
        self.buffer = ""
        self.mode = "text"
        self.strip_role_after_im_start = False
        self.open_tags = {
            "thinking": _THINK_OPEN,
            "thinking_close": _THINK_CLOSE,
            "tool": "<tool_call>",
            "im_start": _IM_START,
            "im_end": _IM_END,
        }
        self.close_tags = {"thinking": _THINK_CLOSE, "tool": "</tool_call>"}
        self.max_tag_length = max(
            len(tag)
            for tag in [
                self.open_tags["thinking"],
                self.open_tags["tool"],
                self.open_tags["im_start"],
                self.open_tags["im_end"],
                self.close_tags["thinking"],
                self.close_tags["tool"],
            ]
        )

    def push(self, chunk: str) -> list[dict]:
        if not chunk:
            return []

        self.buffer += chunk
        return self._drain_buffer(final=False)

    def finish(self) -> list[dict]:
        return self._drain_buffer(final=True)

    def _drain_buffer(self, final: bool) -> list[dict]:
        segments: list[dict] = []

        while True:
            if self.mode == "text":
                if self.strip_role_after_im_start:
                    role_match = re.match(r"^\s*\w+\s*", self.buffer)
                    if role_match:
                        self.buffer = self.buffer[role_match.end() :]
                        self.strip_role_after_im_start = False
                        continue

                    if final:
                        self.strip_role_after_im_start = False
                    else:
                        break

                marker = self._find_next_open_marker()
                if marker is None:
                    if final:
                        if self.buffer:
                            segments.append({"kind": "text", "text": self.buffer})
                            self.buffer = ""
                        break

                    if len(self.buffer) <= self.max_tag_length - 1:
                        break
                    emit = self.buffer[: -(self.max_tag_length - 1)]
                    if emit:
                        segments.append({"kind": "text", "text": emit})
                    self.buffer = self.buffer[-(self.max_tag_length - 1) :]
                    break

                marker_index, marker_type, marker_tag = marker
                if marker_index > 0:
                    segments.append(
                        {"kind": "text", "text": self.buffer[:marker_index]}
                    )
                self.buffer = self.buffer[marker_index + len(marker_tag) :]

                if marker_type == "im_start":
                    self.strip_role_after_im_start = True
                    continue

                if marker_type == "thinking_close":
                    continue

                if marker_type == "im_end":
                    continue

                self.mode = marker_type
                continue

            close_tag = self.close_tags[self.mode]
            close_index = self.buffer.find(close_tag)

            if close_index == -1:
                if self.mode == "tool":
                    if final:
                        tool_segment = _tool_segment_from_payload(self.buffer)
                        if tool_segment is not None:
                            segments.append(tool_segment)
                        else:
                            wrapped = f"<tool_call>{self.buffer}</tool_call>"
                            segments.append({"kind": "text", "text": wrapped})
                        self.buffer = ""
                        self.mode = "text"
                    break

                if final:
                    content = self.buffer
                    self.buffer = ""
                    self.mode = "text"
                    if self.enable_thinking and content:
                        segments.append({"kind": "thinking", "thinking": content})
                    break

                if len(self.buffer) <= len(close_tag) - 1:
                    break

                emit = self.buffer[: -(len(close_tag) - 1)]
                if emit and self.enable_thinking:
                    segments.append({"kind": "thinking", "thinking": emit})
                self.buffer = self.buffer[-(len(close_tag) - 1) :]
                break

            content = self.buffer[:close_index]
            self.buffer = self.buffer[close_index + len(close_tag) :]
            segment_kind = "thinking" if self.mode == "thinking" else "tool"
            self.mode = "text"

            if segment_kind == "thinking":
                if self.enable_thinking and content:
                    segments.append({"kind": "thinking", "thinking": content})
                continue

            tool_segment = _tool_segment_from_payload(content)
            if tool_segment is None:
                wrapped = f"<tool_call>{content}</tool_call>"
                segments.append({"kind": "text", "text": wrapped})
            else:
                segments.append(tool_segment)

        return segments

    def _find_next_open_marker(self):
        candidates = []
        for marker_type, marker_tag in self.open_tags.items():
            index = self.buffer.find(marker_tag)
            if index != -1:
                candidates.append((index, marker_type, marker_tag))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])


def _tool_segment_from_payload(payload: str) -> dict | None:
    try:
        parsed = json.loads(payload)
    except Exception:
        return _tool_segment_from_malformed_payload(payload)

    if not isinstance(parsed, dict):
        return None

    tool_name = parsed.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return None

    tool_input = parsed.get("input")
    if tool_input is None:
        tool_input = parsed.get("arguments")
    if tool_input is None:
        tool_input = {}
    if not isinstance(tool_input, dict):
        return None

    return {
        "kind": "tool_use",
        "name": tool_name,
        "partial_json": json.dumps(tool_input, ensure_ascii=False),
    }


def _tool_segment_from_malformed_payload(payload: str) -> dict | None:
    if not payload:
        return None

    tool_name_match = re.search(r'"name"\s*:\s*"([^"]+)"', payload)
    if not tool_name_match:
        return None

    tool_name = tool_name_match.group(1).strip()
    if not tool_name:
        return None

    # When arguments/input JSON is malformed or truncated, still emit a tool_use
    # block with empty input so the client can request permission and recover.
    return {
        "kind": "tool_use",
        "name": tool_name,
        "partial_json": "{}",
    }


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
        role = msg.role

        if isinstance(msg.content, str):
            messages.append({"role": role, "content": msg.content})
            continue

        content_text = []

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

            elif isinstance(c, ToolUseBlockParam):
                content_text.append(
                    f"<tool_call>{json.dumps({'name': c.name, 'arguments': c.input}, ensure_ascii=False)}</tool_call>"
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
