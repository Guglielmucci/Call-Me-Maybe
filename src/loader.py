"""JSON loading utilities."""
from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)


def load_json_items(filepath: Path, model: type, label: str) -> list[Any]:
    """Load a list of objects from a JSON file, validated with Pydantic model.

    Args:
        filepath: Path to the JSON file.
        model: Pydantic class used for validation.
        label: Descriptive label for error messages.

    Returns:
        List of validated instances.

    Raises:
        SystemExit: On file not found, invalid JSON, or validation error.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        logger.error(
            "%s file not found: %s",
            label,
            filepath
        )
        sys.exit(1)
    except json.JSONDecodeError as exc:
        logger.error(
            "Invalid JSON in %s: %s",
            filepath,
            exc
        )
        sys.exit(1)
    except Exception as exc:
        logger.error(
            "Error reading %s file: %s",
            filepath,
            exc
        )
        sys.exit(1)
    items: list[Any] = []
    for i, obj in enumerate(raw_data, 1):
        try:
            items.append(model(**obj))
        except ValidationError as exc:
            logger.error(
                "Validation error in %s %d: %s",
                label,
                i,
                exc
            )
            sys.exit(1)
    return items
