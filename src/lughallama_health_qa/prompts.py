from __future__ import annotations

from typing import Any


def language_for_subset(subset: str, cfg: dict[str, Any]) -> str:
    return cfg.get("language_names", {}).get(str(subset), str(subset))


def build_prompt(question: str, language: str, prompt_cfg: dict[str, Any]) -> str:
    return prompt_cfg["template"].format(
        question=str(question).strip(),
        language=str(language).strip(),
    )
