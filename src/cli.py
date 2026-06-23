"""Command line interface helpers."""
from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
from pydantic import ValidationError

from .models import AppConfig


def parse_args(argv: Optional[list[str]] = None) -> AppConfig:
    """Parse command‑line arguments and return validated application config.

    Args:
        argv: List of arguments (defaults to ``sys.argv[1:]``).

    Returns:
        An ``AppConfig`` instance with validated values.

    Raises:
        SystemExit: if the arguments fail validation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate structured function calls from natural language prompts."
        )
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=AppConfig.model_fields["functions_definition"].default,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=AppConfig.model_fields["input"].default,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=AppConfig.model_fields["output"].default,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=AppConfig.model_fields["model"].default,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=AppConfig.model_fields["verbose"].default,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help=(
            "Timeout in seconds for each generation "
            "(default: 30, 0=no limit)"
        ),
    )

    args = parser.parse_args(argv)
    try:
        config = AppConfig(**vars(args))
    except ValidationError as e:
        print(f"Argument validation error: {e}", file=sys.stderr)
        sys.exit(1)
    return config


def setup_logging(verbose: bool) -> None:
    """Configure the logging level and format.

    Args:
        verbose: If ``True``, set level to DEBUG and include timestamps.
    """
    level = logging.DEBUG if verbose else logging.INFO
    if verbose:
        fmt = "%(asctime)s [%(levelname)s] %(message)s"
    else:
        fmt = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
