import json

from inference import _remap_unknown_tool_call


def test_remap_listfiles_to_bash_when_available():
    tool_name, partial_json = _remap_unknown_tool_call(
        tool_name="ListFiles",
        partial_json=json.dumps({"directory": "."}),
        available_tool_names={"Bash", "Read"},
    )

    assert tool_name == "Bash"
    payload = json.loads(partial_json)
    assert payload["command"].startswith("ls -la --")
    assert payload["description"] == "List files in directory"


def test_keep_unknown_tool_when_no_mapping_available():
    tool_name, partial_json = _remap_unknown_tool_call(
        tool_name="ListFiles",
        partial_json=json.dumps({"directory": "."}),
        available_tool_names={"Read"},
    )

    assert tool_name == "ListFiles"
    assert json.loads(partial_json)["directory"] == "."


def test_remap_ls_to_bash_when_available():
    tool_name, partial_json = _remap_unknown_tool_call(
        tool_name="ls",
        partial_json=json.dumps({}),
        available_tool_names={"Bash", "Read"},
    )

    assert tool_name == "Bash"
    payload = json.loads(partial_json)
    assert payload["command"] == "ls -la -- ."
    assert payload["description"] == "List files in directory"
