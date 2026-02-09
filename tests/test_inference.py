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

    for i, m in enumerate(params.messages):
        expected_role = m.role

        for j, c in enumerate(m.content):
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

            # elif c.type == "tool_result":
            #     assert result.messages[i]["content"][j]["url"] == c.
            #     expected_role = "tool"

        assert result_messages[i]["role"] == expected_role


def test_parse_claude_tools(params):
    pass


def test_parse_claude_max_tokens():
    pass


def test_build_prompt():
    pass
