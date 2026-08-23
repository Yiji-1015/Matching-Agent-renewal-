from pathlib import Path

from .config import PROMPTS_DIR


def load_prompt(name: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    """Load a role prompt by filename from the prompts directory."""
    return (prompts_dir / name).read_text(encoding="utf-8")
