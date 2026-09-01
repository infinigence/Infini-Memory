"""Test robustness of the parse_merge_groups function."""

import pytest


def test_parse_merge_groups_normal():
    """Test normal JSON input."""
    from infini_memory_classic.memory import parse_merge_groups

    json_str = '{"groups": [{"doc_ids": ["1_2025-01-01_abc", "1_2025-01-02_def"], "reason": "测试分组"}]}'
    result = parse_merge_groups(json_str)

    assert result == {"groups": [{"doc_ids": ["1_2025-01-01_abc", "1_2025-01-02_def"], "reason": "测试分组"}]}


def test_parse_merge_groups_extra_data():
    """Test JSON with extra data (original error scenario)."""
    from infini_memory_classic.memory import parse_merge_groups

    # Simulate LLM returning multiple JSON objects concatenated together
    json_str = '''{"groups": [{"doc_ids": ["1_2025-01-01_abc"], "reason": "测试"}]}
{"groups": []}'''

    result = parse_merge_groups(json_str)

    # Should successfully parse the first JSON object
    assert result == {"groups": [{"doc_ids": ["1_2025-01-01_abc"], "reason": "测试"}]}


def test_parse_merge_groups_markdown_wrap():
    """Test JSON wrapped in markdown code block."""
    from infini_memory_classic.memory import parse_merge_groups

    json_str = '''```json
{"groups": [{"doc_ids": ["1_2025-01-01_abc"], "reason": "测试分组"}]}
```'''
    result = parse_merge_groups(json_str)

    assert result == {"groups": [{"doc_ids": ["1_2025-01-01_abc"], "reason": "测试分组"}]}


def test_parse_merge_groups_truncated():
    """Test truncated JSON (missing closing brackets)."""
    from infini_memory_classic.memory import parse_merge_groups

    json_str = '{"groups": [{"doc_ids": ["1_2025-01-01_abc"], "reason": "测试"'
    result = parse_merge_groups(json_str)

    # Should fix and parse, or return empty result
    assert "groups" in result


def test_parse_merge_groups_empty():
    """Test empty input."""
    from infini_memory_classic.memory import parse_merge_groups

    result = parse_merge_groups("")
    assert result == {"groups": []}


def test_parse_merge_groups_invalid():
    """Test completely invalid input."""
    from infini_memory_classic.memory import parse_merge_groups

    result = parse_merge_groups("this is not json at all")
    assert result == {"groups": []}


def test_parse_merge_groups_with_comments():
    """Test JSON with JavaScript comments."""
    from infini_memory_classic.memory import parse_merge_groups

    json_str = '''{
  // 这是一个注释
  "groups": [{"doc_ids": ["1_2025-01-01_abc"], "reason": "测试"}]
}'''
    result = parse_merge_groups(json_str)

    assert result == {"groups": [{"doc_ids": ["1_2025-01-01_abc"], "reason": "测试"}]}


def test_normalize_merge_groups_data():
    """Test data normalization function."""
    from infini_memory_classic.memory import _normalize_merge_groups_data

    # Test normal data
    data = {"groups": [{"doc_ids": ["1", "2"], "reason": "测试"}]}
    result = _normalize_merge_groups_data(data)
    assert result == {"groups": [{"doc_ids": ["1", "2"], "reason": "测试"}]}

    # Test numeric doc_ids
    data = {"groups": [{"doc_ids": [1, 2], "reason": "测试"}]}
    result = _normalize_merge_groups_data(data)
    assert result == {"groups": [{"doc_ids": ["1", "2"], "reason": "测试"}]}

    # Test invalid data
    data = {"groups": "not a list"}
    result = _normalize_merge_groups_data(data)
    assert result == {"groups": []}

    # Test empty doc_ids
    data = {"groups": [{"doc_ids": [], "reason": "测试"}]}
    result = _normalize_merge_groups_data(data)
    assert result == {"groups": []}
