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


class Parser(BaseChatParser):
    def parse_chat_params(self, params: ClaudeMessageParams) -> ChatParams:
        return ChatParams(
            messages=_parse_messages(params),
            tools=_parse_tools(params),
            enable_thinking=_parse_thinking(params),
            max_tokens=params.max_tokens,
            # sampler_params=_parse_sampler(params),
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


# TODO:
# def _parse_sampler(params: ClaudeMessageParams):
#     temp = (
#         params.temperature
#         if params.temperature is not None
#         else settings.default_temperature
#     )
#     sampler_params = {"temp": temp}

#     if params.top_k is not None:
#         sampler_params["top_k"] = params.top_k

#     if params.top_p is not None:
#         sampler_params["top_p"] = params.top_p

#     return sampler_params
