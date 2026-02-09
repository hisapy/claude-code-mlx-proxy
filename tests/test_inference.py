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

    assert len(params.messages)
    non_system_messages = enumerate(params.messages, len(params.system))
    for i, msg in non_system_messages:
        assert result.messages[i]["role"] == msg.role

        for j, c in enumerate(msg.content):
            assert result.messages[i]["content"][j]["type"] == c.type
            if c.type == "text":
                assert result.messages[i]["content"][j]["text"] == c.text

        # if {"type": "tool_result"} in [{"type": c.type} for c in msg.content]:
        #     assert result.messages[i] == {"role": msg.role, "content": msg.content}


# def test_parse_claude_tools():
#     pass


# def test_parse_claude_max_tokens():
#     pass


# def test_build_prompt():
#     pass
