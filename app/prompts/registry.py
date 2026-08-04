from pathlib import Path
import yaml

_DEFAULT_BASE = Path(__file__).parent


class PromptRegistry:
    def __init__(self, base_dir: Path = _DEFAULT_BASE) -> None:
        self._base = base_dir
        self._cache: dict[str, dict] = {}

    def load(self, prompt_type: str, variant: str) -> dict:
        key = f"{prompt_type}/{variant}"
        if key not in self._cache:
            path = self._base / prompt_type / f"{variant}.yaml"
            with open(path, encoding="utf-8") as f:
                self._cache[key] = yaml.safe_load(f)
        return self._cache[key]

    def format(self, prompt_type: str, variant: str, **kwargs) -> tuple[str, float]:
        """Return (formatted_prompt_string, temperature)."""
        config = self.load(prompt_type, variant)

        few_shots_text = ""
        if "few_shots" in config:
            parts = [
                f"INPUT:\n{ex['input'].strip()}\n\nOUTPUT:\n{ex['output'].strip()}"
                for ex in config["few_shots"]
            ]
            few_shots_text = "\n\n---\n\n".join(parts)

        prompt = config["template"].format(
            system=config.get("system", ""),
            few_shots_formatted=few_shots_text,
            **kwargs,
        )
        return prompt, float(config.get("temperature", 0.1))


registry = PromptRegistry()
