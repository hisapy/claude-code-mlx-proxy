import pytest
import json
import pathlib

from inference import ChatParams, parse_claude_message_params, build_prompt
from schemas import ClaudeMessageParams


def _get_json_files():
    data_path = pathlib.Path(__file__).parent / "fixtures"
    return list(data_path.glob("*.json"))


@pytest.fixture(autouse=True, params=_get_json_files(), ids=lambda f: f.name)
def params(request):
    with open(request.param, "r") as f:
        claude_req_body = json.load(f)

    return ClaudeMessageParams(**claude_req_body)


def test_parse_claude_system_prompt(params):
    result: ChatParams = parse_claude_message_params(params)

    assert len(params.system)
    for i, msg in enumerate(params.system):
        assert result.messages[i] == {
            "role": "system",
            "content": [{"type": "text", "text": msg.text}],
        }


def test_parse_claude_messages(params):
    result: ChatParams = parse_claude_message_params(params)

    # In MLX(transformers) format the system prompt is at the beginning of the conversation
    result_messages = result.messages[len(params.system) :]

    assert len(result_messages)

    for i, msg in enumerate(params.messages):
        expected_role = msg.role

        for j, c in enumerate(msg.content):
            if c.type == "text":
                assert result_messages[i]["content"][j] == {
                    "type": "text",
                    "text": c.text,
                }

            elif c.type == "image":
                assert result.messages[i]["content"][j] == {
                    "type": "image",
                    "url": c.source,
                }

            elif c.type == "tool_result":
                # NOTICE: the role expected by Transformers and that content must be always a string
                assert isinstance(str, result.messages[i]["content"][j])
                expected_role = "tool"

        assert result_messages[i]["role"] == expected_role


def test_parse_claude_tools(params):
    result: ChatParams = parse_claude_message_params(params)

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
    result: ChatParams = parse_claude_message_params(params)

    if params.thinking and params.thinking["type"] in ["enabled", "adaptive"]:
        assert result.enable_thinking
    else:
        assert not result.enable_thinking


def test_parse_sampler_config(params):
    result: ChatParams = parse_claude_message_params(params)

    # Claude Code always send temperature but top_k and top_p depend on the specific request
    assert result.sampler_params["temp"] == params.temperature
    assert result.sampler_params.get("top_k") == params.top_k
    assert result.sampler_params.get("top_p") == params.top_p


def test_build_prompt():
    pass
