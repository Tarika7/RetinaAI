from pathlib import Path
from typing import Union


def generate_unique_filename(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def ensure_directory(path: Union[str, Path]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
