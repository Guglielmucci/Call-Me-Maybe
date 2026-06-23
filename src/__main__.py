"""Entry point for the function calling tool with constrained decoding."""
from __future__ import annotations

import sys
import json
import logging
import time
from pathlib import Path
from typing import Optional

from .cli import parse_args, setup_logging
from .loader import load_json_items
from .timeout import time_limit
from .model_factory import build_model
from .models import FunctionDef, TestItem, OutputItem
from .token_utils import load_vocab
from .tokenizer import CustomTokenizer
from .decoder import ConstrainedDecoder


# ------------------------------------------------------------------
# Optional real model backend (llm_sdk).  Falls back to None if the
# package is not installed.
# ------------------------------------------------------------------
Small_LLM_Model: Optional[type] = None
try:
    import llm_sdk
    Small_LLM_Model = getattr(llm_sdk, 'Small_LLM_Model', None)
except ImportError:
    pass

logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the function calling tool with constrained decoding.

    Loads function definitions and test prompts from JSON files,
    initialises the model, vocabulary, custom tokenizer and constrained
    decoder. Iterates over each prompt, performs generation with timeout
    handling, collects results and writes them to an output file. If
    requested, prints a performance summary.

    Args:
        argv: Command‑line arguments. If ``None``, ``sys.argv[1:]`` is
            used. The arguments are parsed by :func:`parse_args`.

    Returns:
        ``None``. Results are written to the output file specified in the
        configuration.

    Note:
        - Errors during loading, initialisation or generation are logged
          internally; for a failing prompt a result with
          ``name="__error__"`` is recorded.
        - If a test exceeds the configured timeout, a ``TimeoutError`` is
          caught and an ``__error__`` result is inserted.
        - Internal keys (e.g. ``_error``) are stripped before building
          the ``OutputItem``.
    """
    config = parse_args(argv)
    setup_logging(config.verbose)
    # -----------------------------------------------------------------
    # Load function definitions
    # -----------------------------------------------------------------
    logger.debug(
        "Loading function definitions from %s", config.functions_definition
    )
    functions: list[FunctionDef] = load_json_items(
        config.functions_definition,
        FunctionDef,
        "function"
    )
    # -----------------------------------------------------------------
    # Load test prompts
    # -----------------------------------------------------------------
    logger.debug(
        "Loading prompts from %s",
        config.input
    )
    tests: list[TestItem] = load_json_items(
        config.input, TestItem, "test"
    )
    # -----------------------------------------------------------------
    # Initialise model
    # -----------------------------------------------------------------
    try:
        model = build_model(config.model)
    except Exception as exc:
        logger.error(
            "Failed to initialise model '%s': %s",
            config.model,
            exc
        )
        sys.exit(1)
    # -----------------------------------------------------------------
    # Load vocabulary
    # -----------------------------------------------------------------
    try:
        vocab_path_str = model.get_path_to_vocab_file()
    except Exception as exc:
        logger.error(
            "Failed to get vocabulary file path: %s",
            exc
        )
        sys.exit(1)
    vocab_path = Path(vocab_path_str)
    if not vocab_path.exists():
        logger.error(
            "Vocabulary file not found: %s",
            vocab_path_str
        )
        sys.exit(1)
    try:
        vocab: dict[int, str] = load_vocab(str(vocab_path))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(
            "Error in vocabulary file: %s",
            exc
        )
        sys.exit(1)
    except Exception as exc:
        logger.error(
            "Unexpected error loading vocabulary: %s",
            exc
        )
        sys.exit(1)
    # -----------------------------------------------------------------
    # Tokenizer uses the already loaded vocabulary
    # -----------------------------------------------------------------
    tokenizer = CustomTokenizer(
        vocab
    )
    decoder = ConstrainedDecoder(
        model,
        vocab,
        functions,
        tokenizer
    )
    # -----------------------------------------------------------------
    #  Info Generation
    # -----------------------------------------------------------------
    results: list[OutputItem] = []
    total_time = 0.0
    max_prompt_len = 80
    for idx, test in enumerate(tests, 1):
        short_prompt = (
            test.prompt
            if len(test.prompt) <= max_prompt_len
            else test.prompt[:max_prompt_len] + "..."
        )
        logger.info(
            "[%d/%d] Prompt: %s",
            idx,
            len(tests),
            short_prompt
        )
        start = time.time()
        try:
            if config.timeout > 0:
                with time_limit(config.timeout):
                    out = decoder.generate(test.prompt)
            else:
                out = decoder.generate(test.prompt)
            elapsed = time.time() - start
            total_time += elapsed
            # Remove internal keys before building the output item ----
            out.pop("_error", None)
            results.append(OutputItem(prompt=test.prompt, **out))
            logger.info(
                "    -> %s (%.2fs)",
                out.get("name", "?"),
                elapsed
            )
        except TimeoutError:
            elapsed = time.time() - start
            total_time += elapsed
            logger.error(
                "Timeout for prompt '%s' (%.1fs)",
                short_prompt,
                elapsed
            )
            results.append(
                OutputItem(
                    prompt=test.prompt,
                    name="__error__",
                    parameters={}
                )
            )
        except Exception as exc:
            elapsed = time.time() - start
            total_time += elapsed
            logger.error(
                "Error during generation for '%s': %s",
                short_prompt,
                exc
            )
            results.append(
                OutputItem(
                    prompt=test.prompt,
                    name="__error__",
                    parameters={},
                )
            )
    # -----------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------
    logger.info(
        "Writing results to %s",
        config.output
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump(exclude_none=True) for r in results],
            f,
            indent=2,
            ensure_ascii=False,
        )
    # -----------------------------------------------------------------
    # if Verbose write result
    # -----------------------------------------------------------------
    if config.verbose:
        logger.info("\n========= Performance summary =========")
        logger.info("Prompts processed: %d", len(tests))
        logger.info("Total time: %.2f seconds", total_time)
        if tests:
            logger.info(
                "Average per prompt: %.2f seconds",
                total_time / len(tests),
            )
    # -----------------------------------------------------------------
    #  Result Generated
    # -----------------------------------------------------------------
    logger.info(
        "Completed: %d results generated.", len(results)
    )


if __name__ == "__main__":
    main()
