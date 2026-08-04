import pytest
from pathlib import Path
import yaml
from app.prompts.registry import PromptRegistry


@pytest.fixture
def registry_with_fixtures(tmp_path):
    structuring_dir = tmp_path / "structuring"
    structuring_dir.mkdir()
    yaml_content = {
        "name": "test_v1",
        "version": "1.0",
        "temperature": 0.1,
        "system": "You are a test system.",
        "few_shots": [
            {"input": "sample ocr text", "output": '{"document_type": "PASSPORT"}'}
        ],
        "template": "{system}\nEXAMPLES:\n{few_shots_formatted}\nNOW ANALYZE:\n{ocr_text}",
    }
    (structuring_dir / "test_v1.yaml").write_text(yaml.dump(yaml_content))
    reg = PromptRegistry(base_dir=tmp_path)
    return reg


def test_load_prompt(registry_with_fixtures):
    config = registry_with_fixtures.load("structuring", "test_v1")
    assert config["name"] == "test_v1"
    assert config["temperature"] == 0.1


def test_format_prompt_includes_ocr_text(registry_with_fixtures):
    prompt, temp = registry_with_fixtures.format("structuring", "test_v1", ocr_text="my ocr text")
    assert "my ocr text" in prompt
    assert temp == 0.1


def test_format_prompt_includes_few_shots(registry_with_fixtures):
    prompt, _ = registry_with_fixtures.format("structuring", "test_v1", ocr_text="anything")
    assert "sample ocr text" in prompt
    assert "PASSPORT" in prompt
