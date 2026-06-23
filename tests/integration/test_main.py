"""Integration tests for the main pipeline.

These tests exercise the ``main()`` function with mocked external
dependencies (model factory, decoder) and temporary files.  Each test
focuses on a single aspect of the orchestration logic: successful
generation, missing input files, model initialisation failures, timeouts,
and unexpected exceptions during generation.

All tests use the shared fixtures from ``conftest.py`` (vocab_file,
func_file, make_input_file) to avoid duplicating setup code.
"""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Callable, List
from unittest.mock import MagicMock, patch

import pytest

from src.__main__ import main


# ========================================================================
#   Local helper
# ========================================================================
def run_main(args: List[str]) -> None:
    """Invoke ``main()`` as if ``sys.argv`` contained the given arguments.

    The first element (program name) is automatically prepended.

    Args:
        args: command‑line arguments without the program name.
    """
    with patch("sys.argv", ["prog"] + args):
        main()


# ========================================================================
#   Mock fixtures
# ========================================================================
@pytest.fixture
def mock_build_model() -> Generator[MagicMock, None, None]:
    """Patch ``src.__main__.build_model`` and return the mock.

    The returned mock is configured by each test to simulate different
    model behaviours (success, failure, etc.).
    """
    with patch("src.__main__.build_model") as mock:
        yield mock


@pytest.fixture
def mock_decoder_generate() -> Generator[MagicMock, None, None]:
    """Patch ``src.decoder.ConstrainedDecoder.generate`` and return the mock.

    Tests can set ``return_value`` or ``side_effect`` to simulate the
    generation result without interacting with the real model.
    """
    with patch("src.decoder.ConstrainedDecoder.generate") as mock:
        yield mock


# ========================================================================
#   Happy path test
# ========================================================================
def test_main_happy_path(
    mock_build_model: MagicMock,
    mock_decoder_generate: MagicMock,
    vocab_file: Path,
    func_file: Path,
    make_input_file: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """A complete run with valid inputs and a cooperative decoder.

    The decoder mock returns a predefined result; the test checks that
    the output file is written correctly, containing the original prompt
    and the generated function call.

    Args:
        mock_build_model: mock of the model factory.
        mock_decoder_generate: mock of the decoder's ``generate`` method.
        vocab_file: temporary vocabulary JSON file.
        func_file: temporary function definitions JSON file.
        make_input_file: helper to create a prompts JSON file.
        tmp_path: pytest temporary directory.
    """
    input_file = make_input_file("Calculate 42")

    # Configure the mock model
    model_mock = mock_build_model.return_value
    model_mock.get_path_to_vocab_file.return_value = str(vocab_file)
    vocab = json.loads(vocab_file.read_text())
    model_mock.get_logits_from_input_ids.return_value = [0.0] * len(vocab)

    # Predetermined decoder result
    expected_result = {
        "name": "fn_add_numbers",
        "parameters": {"a": 42.0, "b": 3.14},
    }
    mock_decoder_generate.return_value = expected_result

    output_file = tmp_path / "output.json"

    run_main([
        "--functions_definition", str(func_file),
        "--input", str(input_file),
        "--output", str(output_file),
    ])

    # Verify the output
    assert output_file.exists(), "Output file was not created"
    results = json.loads(output_file.read_text())
    assert len(results) == 1, "Exactly one result expected"
    assert results[0]["prompt"] == "Calculate 42"
    assert results[0]["name"] == "fn_add_numbers"
    assert results[0]["parameters"] == {"a": 42.0, "b": 3.14}
    mock_decoder_generate.assert_called_once()


# ========================================================================
#   Error path tests
# ========================================================================
def test_main_missing_functions_file(
    make_input_file: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """If the function definitions file does not exist, the program exits.

    Args:
        make_input_file: helper to create a prompts JSON file.
        tmp_path: temporary directory.
    """
    input_file = make_input_file("test")
    output_file = tmp_path / "output.json"
    missing_func = "/nonexistent/funcs.json"

    with pytest.raises(SystemExit) as exc_info:
        run_main([
            "--functions_definition", missing_func,
            "--input", str(input_file),
            "--output", str(output_file),
        ])
    assert exc_info.value.code == 1


def test_main_missing_input_file(
    func_file: Path,
    tmp_path: Path,
) -> None:
    """If the input prompts file does not exist, the program exits.

    Args:
        func_file: valid function definitions file (shared fixture).
        tmp_path: temporary directory.
    """
    output_file = tmp_path / "output.json"
    missing_input = "/nonexistent/input.json"

    with pytest.raises(SystemExit) as exc_info:
        run_main([
            "--functions_definition", str(func_file),
            "--input", missing_input,
            "--output", str(output_file),
        ])
    assert exc_info.value.code == 1


def test_main_model_init_failure(
    mock_build_model: MagicMock,
    func_file: Path,
    make_input_file: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """If ``build_model`` raises exception, the program logs error and exits.

    Args:
        mock_build_model: mock that will raise an exception.
        func_file: valid function definitions file.
        make_input_file: helper for input files.
        tmp_path: temporary directory.
    """
    mock_build_model.side_effect = RuntimeError("Model not available")
    input_file = make_input_file("test")
    output_file = tmp_path / "output.json"

    with pytest.raises(SystemExit) as exc_info:
        run_main([
            "--functions_definition", str(func_file),
            "--input", str(input_file),
            "--output", str(output_file),
        ])
    assert exc_info.value.code == 1


def test_main_timeout_per_prompt(
    mock_build_model: MagicMock,
    mock_decoder_generate: MagicMock,
    vocab_file: Path,
    func_file: Path,
    make_input_file: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """When a prompt times out, the program writes error entry and continues.

    The decoder mock throws ``TimeoutError`` for the first prompt and
    returns a normal result for the second.

    Args:
        mock_build_model: mock of the model factory.
        mock_decoder_generate: mock of the decoder's ``generate`` method.
        vocab_file: temporary vocabulary file.
        func_file: function definitions file.
        make_input_file: helper for input files.
        tmp_path: temporary directory.
    """
    # Configure model mock
    model_mock = mock_build_model.return_value
    model_mock.get_path_to_vocab_file.return_value = str(vocab_file)
    vocab = json.loads(vocab_file.read_text())
    model_mock.get_logits_from_input_ids.return_value = [0.0] * len(vocab)

    # First call raises TimeoutError, second succeeds
    mock_decoder_generate.side_effect = [
        TimeoutError("time limit exceeded"),
        {"name": "fn_add_numbers", "parameters": {"a": 1.0, "b": 2.0}},
    ]

    # Prepare input with two prompts
    input_file = make_input_file("prompt 1")
    prompts = [{"prompt": "prompt 1"}, {"prompt": "prompt 2"}]
    input_file.write_text(json.dumps(prompts))

    output_file = tmp_path / "output.json"

    run_main([
        "--functions_definition", str(func_file),
        "--input", str(input_file),
        "--output", str(output_file),
    ])

    results = json.loads(output_file.read_text())
    assert len(results) == 2
    assert results[0]["name"] == "__error__"
    assert results[1]["name"] == "fn_add_numbers"


def test_main_exception_during_generation(
    mock_build_model: MagicMock,
    mock_decoder_generate: MagicMock,
    vocab_file: Path,
    func_file: Path,
    make_input_file: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """A generic exception during generation produces an error.

    Args:
        mock_build_model: mock of the model factory.
        mock_decoder_generate: mock of the decoder's ``generate`` method.
        vocab_file: temporary vocabulary file.
        func_file: function definitions file.
        make_input_file: helper for input files.
        tmp_path: temporary directory.
    """
    model_mock = mock_build_model.return_value
    model_mock.get_path_to_vocab_file.return_value = str(vocab_file)
    vocab = json.loads(vocab_file.read_text())
    model_mock.get_logits_from_input_ids.return_value = [0.0] * len(vocab)

    mock_decoder_generate.side_effect = [
        ValueError("token not allowed"),
        {"name": "fn_greet", "parameters": {"name": "Jane"}},
    ]

    input_file = make_input_file("prompt 1")
    prompts = [{"prompt": "prompt 1"}, {"prompt": "prompt 2"}]
    input_file.write_text(json.dumps(prompts))
    output_file = tmp_path / "output.json"

    run_main([
        "--functions_definition", str(func_file),
        "--input", str(input_file),
        "--output", str(output_file),
    ])

    results = json.loads(output_file.read_text())
    assert len(results) == 2
    assert results[0]["name"] == "__error__"
    assert results[1]["name"] == "fn_greet"
