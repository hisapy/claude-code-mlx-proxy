import pytest
import json
import pathlib

from inference import ChatParams, parse_claude_message_params, build_prompt
from schemas import ClaudeMessageParams


def _get_json_files():
    data_path = pathlib.Path(__file__).parent / "fixtures"
    return list(data_path.glob("*.json"))


@pytest.fixture
def claude_req_body(request):
    with open(request.param, "r") as f:
        return json.load(f)


@pytest.mark.parametrize(
    "claude_req_body", _get_json_files(), indirect=True, ids=lambda f: f.name
)
def test_parse_claude_system_messages(claude_req_body):
    params = ClaudeMessageParams(**claude_req_body)

    result: ChatParams = parse_claude_message_params(params)

    assert len(params.system)
    for i, msg in enumerate(params.system):
        assert result.messages[i] == {"role": "system", "content": msg.text}


def test_parse_claude_tools():
    pass


def test_parse_claude_max_tokens():
    pass


def test_build_prompt():
    pass
