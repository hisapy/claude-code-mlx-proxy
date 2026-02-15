from typing import Any, Dict

from base_chat_parser import BaseChatParser
from claude_schemas import (
    # Request models
    ClaudeMessageParams,
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
        return ChatParams(
            messages=_parse_messages(params),
            tools=_parse_tools(params),
            enable_thinking=enable_thinking,
            max_tokens=params.max_tokens,
            sampler_params=_parse_sampler(params, enable_thinking),
            request_model=params.model,
        )

    def format_response(self, raw_output: str) -> Dict[str, Any]:
        # TODO: implement
        return raw_output


def _parse_messages(params: ClaudeMessageParams):
    # Put system prompt at the beginning of the chat
    # See https://huggingface.co/docs/transformers/en/chat_templating#using-applychattemplate
    # In the case of qwen3 we have to put all the system messages in messages[0]
    return [
        _parse_system_prompt(params.system),
        *_parse_conversation(params.messages),
    ]


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
            if isinstance(c, TextBlockParam):
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
