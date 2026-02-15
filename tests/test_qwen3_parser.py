import pytest
import json
import pathlib

from mlx_schemas import ChatParams
from claude_schemas import ClaudeMessageParams

from adapters.qwen3 import Parser

parser = Parser()


def _get_json_files():
    data_path = pathlib.Path(__file__).parent / "fixtures"
    return list(data_path.glob("*.json"))


@pytest.fixture(autouse=True, params=_get_json_files(), ids=lambda f: f.name)
def params(request):
    with open(request.param, "r") as f:
        claude_req_body = json.load(f)

    return ClaudeMessageParams(**claude_req_body)


def test_parse_claude_system_prompt(params):
    result: ChatParams = parser.parse_chat_params(params)

    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == "\n\n".join(
        msg.text for msg in params.system
    )


def test_parse_claude_messages(params):
    result: ChatParams = parser.parse_chat_params(params)

    # The parser combines all system messages into one system message at index 0
    # So we skip only the first message (the combined system message)
    result_messages = result.messages[1:]

    assert len(result_messages) == len(params.messages)

    for i, msg in enumerate(params.messages):
        expected_role = msg.role

        # Since content is a string in Message class, we need to check it as a string
        # The parser joins all content blocks with "\n\n"
        expected_content_parts = []

        for c in msg.content:
            if c.type == "text":
                expected_content_parts.append(c.text)
            elif c.type == "image":
                # Check that image content is handled (converted to text representation)
                expected_content_parts.append(
                    f"[Image Context: {c.source:50}] ... (Omitted: Text-only mode)"
                )
            elif c.type == "tool_result":
                # Tool results change the expected role
                expected_role = "tool"
                if isinstance(c.content, str):
                    expected_content_parts.append(
                        f"<tool_response>{c.content}</tool_response>"
                    )
                else:
                    # Handle TextBlockParam or ImageBlockParam content
                    tool_content = (
                        c.content.text
                        if hasattr(c.content, "text")
                        else c.content.source
                    )
                    expected_content_parts.append(
                        f"<tool_response>{tool_content}</tool_response>"
                    )

        expected_content = "\n\n".join(expected_content_parts)
        assert result_messages[i]["content"] == expected_content
        assert result_messages[i]["role"] == expected_role


def test_parse_claude_tools(params):
    result: ChatParams = parser.parse_chat_params(params)

    if not params.tools:
        assert result.tools == None
    else:
        for i, tool in enumerate(params.tools):
            assert result.tools[i] == {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }


def test_parse_enable_thinking(params):
    result: ChatParams = parser.parse_chat_params(params)

    if params.thinking and params.thinking["type"] in ["enabled", "adaptive"]:
        assert result.enable_thinking
    else:
        assert not result.enable_thinking


def test_parse_sampler_config(params):
    from adapters.qwen3 import QWEN3_THINKING_SAMPLER, QWEN3_NON_THINKING_SAMPLER

    result: ChatParams = parser.parse_chat_params(params)

    defaults = (
        QWEN3_THINKING_SAMPLER if result.enable_thinking else QWEN3_NON_THINKING_SAMPLER
    )

    # Temperature: always Qwen3 default (Claude Code always sends temperature,
    # which would break Qwen3's recommended config)
    assert result.sampler_params["temp"] == defaults["temp"]

    # top_k: use request value if provided, otherwise Qwen3 default
    expected_top_k = params.top_k if params.top_k is not None else defaults["top_k"]
    assert result.sampler_params["top_k"] == expected_top_k

    # top_p: use request value if provided, otherwise Qwen3 default
    expected_top_p = params.top_p if params.top_p is not None else defaults["top_p"]
    assert result.sampler_params["top_p"] == expected_top_p

    # min_p: always Qwen3 default
    assert result.sampler_params["min_p"] == defaults["min_p"]


# TODO:
# def test_build_prompt():
#     pass
