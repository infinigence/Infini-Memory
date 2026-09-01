"""Repository-level pytest collection configuration."""

# The AI Demo backend is a standalone project with its own pyproject.toml and
# dependency environment. Its tests are run from ai_demo/backend instead of as
# part of the root project's test suite. The SeaweedFS script is a manually
# parameterized connectivity check rather than a pytest test module.
collect_ignore = ["ai_demo/backend/tests", "scripts/test_seaweedfs.py"]
