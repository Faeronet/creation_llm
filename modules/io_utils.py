"""
I/O utilities: JSONL, text files, YAML. UTF-8 encoding and error handling.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import yaml

from modules.exceptions import DataNotFoundError
from modules.logger import get_logger

logger = get_logger(__name__)

ENCODING = "utf-8"


def read_text(path: Union[str, Path]) -> str:
    """
    Read a text file as UTF-8.

    Args:
        path: Path to the file.

    Returns:
        File contents as string.

    Raises:
        DataNotFoundError: If file does not exist or read fails.
    """
    p = Path(path)
    if not p.exists():
        raise DataNotFoundError(f"File not found: {p}")
    try:
        return p.read_text(encoding=ENCODING)
    except Exception as e:
        raise DataNotFoundError(f"Failed to read {p}: {e}") from e


def write_text(path: Union[str, Path], content: str) -> None:
    """Write string to file as UTF-8. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=ENCODING)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read a JSONL file. One JSON object per line.

    Args:
        path: Path to the file.

    Returns:
        List of parsed JSON objects.

    Raises:
        DataNotFoundError: If file does not exist or parse fails.
    """
    p = Path(path)
    if not p.exists():
        raise DataNotFoundError(f"File not found: {p}")
    out: List[Dict[str, Any]] = []
    try:
        with open(p, "r", encoding=ENCODING) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    except json.JSONDecodeError as e:
        raise DataNotFoundError(f"Invalid JSON at {p} line {i + 1}: {e}") from e
    except Exception as e:
        raise DataNotFoundError(f"Failed to read {p}: {e}") from e
    return out


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Iterate over JSONL file without loading all into memory."""
    p = Path(path)
    if not p.exists():
        raise DataNotFoundError(f"File not found: {p}")
    with open(p, "r", encoding=ENCODING) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Union[str, Path], records: List[Dict[str, Any]]) -> None:
    """Write list of dicts as JSONL. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding=ENCODING) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path: Union[str, Path]) -> Any:
    """Read a single JSON file."""
    p = Path(path)
    if not p.exists():
        raise DataNotFoundError(f"File not found: {p}")
    with open(p, "r", encoding=ENCODING) as f:
        return json.load(f)


def write_json(path: Union[str, Path], data: Any) -> None:
    """Write JSON file. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding=ENCODING) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Read YAML file."""
    p = Path(path)
    if not p.exists():
        raise DataNotFoundError(f"File not found: {p}")
    with open(p, "r", encoding=ENCODING) as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Union[str, Path], data: Dict[str, Any]) -> None:
    """Write YAML file. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding=ENCODING) as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)


def ensure_data_file(path: Path, description: str) -> None:
    """
    Check that a data file exists; if not, log and raise DataNotFoundError
    with a message pointing to README.
    """
    if not path.exists():
        logger.error("Missing data file: %s (%s)", path, description)
        raise DataNotFoundError(
            f"Required file not found: {path}. Please place the dataset files in data/ as described in README.md."
        )
