import pytest

from tarantula.llm import _parse_json_content


def test_parse_clean_json():
    assert _parse_json_content('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_tolerates_trailing_extra_data():
    # Regression: the model returned a valid object followed by extra content on
    # a second line, which bare json.loads rejected with "Extra data: line 2".
    content = '{"action": "grep", "pattern": "x"}\n{"action": "answer"}'
    assert _parse_json_content(content) == {"action": "grep", "pattern": "x"}


def test_parse_strips_markdown_json_fence():
    content = '```json\n{"value": 42}\n```'
    assert _parse_json_content(content) == {"value": 42}


def test_parse_strips_plain_fence():
    content = '```\n{"value": 42}\n```'
    assert _parse_json_content(content) == {"value": 42}


def test_parse_handles_surrounding_whitespace():
    assert _parse_json_content('  \n {"a": 1}\n ') == {"a": 1}


def test_parse_raises_on_non_json():
    with pytest.raises(ValueError):
        _parse_json_content("not json at all")
