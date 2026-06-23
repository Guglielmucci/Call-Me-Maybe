"""Shared fixtures for the entire test suite."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
from unittest.mock import MagicMock
from _pytest.terminal import TerminalReporter
from _pytest.config import Config

from src.models import FunctionDef
from src.tokenizer import CustomTokenizer
from typing import TypedDict


class TestCounts(TypedDict):
    """Count of passed and failed tests for a group."""

    passed: int
    failed: int


@pytest.fixture
def sample_functions_def() -> List[Dict[str, Any]]:
    """Sample function definitions as a list of dictionaries.

    Returns:
        Three example functions (addition, greeting, string reversal)
        with parameters and return types.
    """
    return [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers",
            "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
            "returns": {"type": "number"},
        },
        {
            "name": "fn_greet",
            "description": "Greet a person",
            "parameters": {"name": {"type": "string"}},
            "returns": {"type": "string"},
        },
        {
            "name": "fn_reverse_string",
            "description": "Reverse a string",
            "parameters": {"s": {"type": "string"}},
            "returns": {"type": "string"},
        },
    ]


@pytest.fixture
def vocab_small() -> Dict[int, str]:
    """Reduced yet complete vocabulary for decoder and schema tests.

    Token IDs are listed in increasing order to simplify test sequences.
    Includes special characters for space (Ġ) and newline (Ċ) as used
    by the real tokenizer.

    Returns:
        Mapping from token ID to string.
    """
    return {
        # JSON delimiters
        0: "{",
        1: "}",
        2: '"',
        3: ":",
        4: ",",
        # Recurring keys in function definitions
        5: '"name"',
        6: '"parameters"',
        7: '"description"',
        8: '"returns"',
        9: '"type"',
        # Types
        10: "number",
        11: "string",
        # Full function names
        12: "fn_add_numbers",
        13: "fn_greet",
        14: "fn_reverse_string",
        # Parameter names
        15: "a",
        16: "b",
        17: "name",
        # Literal values used in tests
        18: "true",
        19: "false",
        20: "42",
        21: "John",
        22: "3.14",
        # Special characters for space and newline
        # (tokenizer preprocessing)
        23: "Ġ",
        24: "Ċ",
    }


@pytest.fixture
def mock_model(vocab_small: Dict[int, str]) -> MagicMock:
    """Return zero logits and a fake vocabulary path.

    Args:
        vocab_small: Reduced vocabulary fixture.

    Returns:
        Mock object with ``get_logits_from_input_ids`` and
        ``get_path_to_vocabulary_json`` configured.
    """
    model = MagicMock()
    # All-equal logits (neutral model) for deterministic sampling tests
    model.get_logits_from_input_ids.return_value = [0.0] * len(vocab_small)
    model.get_path_to_vocabulary_json.return_value = "fake_path"
    return model


@pytest.fixture
def mock_tokenizer(vocab_small: Dict[int, str]) -> MagicMock:
    """Mock tokenizer with longest match encoding and space, newline handling.

    Emulates the real CustomTokenizer: converts before encoding, and reverses
    the transformation on decoding.

    Args:
        vocab_small: Reduced vocabulary fixture.

    Returns:
        Mock object with ``encode`` and ``decode`` side_effect implementations.
    """
    tokenizer = MagicMock(spec=CustomTokenizer)
    # Bidirectional mappings ID ↔ token
    token_to_id = {v: k for k, v in vocab_small.items()}
    id_to_token = vocab_small

    def encode(text: str) -> List[int]:
        """Longest match enc with fallback to token 0 for unknown symbols."""
        # Preprocessing: spaces and newlines become Ġ and Ċ
        text = text.replace(" ", "Ġ").replace("\n", "Ċ")
        ids: List[int] = []
        i = 0
        while i < len(text):
            best_len = 0
            best_id = None
            # Look for the longest matching token starting at position i
            for token, tid in token_to_id.items():
                if text.startswith(token, i) and len(token) > best_len:
                    best_len = len(token)
                    best_id = tid
            if best_id is not None:
                ids.append(best_id)
                i += best_len
            else:
                # Out‑of‑vocabulary character: dummy token (0) to avoid
                # blocking the tests
                ids.append(0)
                i += 1
        return ids

    def decode(ids: List[int]) -> str:
        """Decode restoring spaces and newlines."""
        # Rebuild the string and reverse the preprocessing
        return "".join(
            id_to_token[t] for t in ids
        ).replace("Ġ", " ").replace("Ċ", "\n")
    # Assign functions as side_effect to keep the mock spec‑compliant
    tokenizer.encode = MagicMock(side_effect=encode)
    tokenizer.decode = MagicMock(side_effect=decode)
    return tokenizer


@pytest.fixture
def sample_funcs(
    sample_functions_def: List[Dict[str, Any]],
) -> List[FunctionDef]:
    """List of FunctionDef (Pydantic) objects built from sample definitions.

    Args:
        sample_functions_def: Fixture containing raw dictionaries.

    Returns:
        Validated objects used in validation and parsing tests.
    """
    return [FunctionDef(**f) for f in sample_functions_def]


@pytest.fixture
def func_file(
    tmp_path: Path, sample_functions_def: List[Dict[str, Any]]
) -> Path:
    """Temporary JSON file holding the function definitions.

    Args:
        tmp_path: Built‑in pytest temporary directory fixture.
        sample_functions_def: Fixture with the data.

    Returns:
        Path to the ``functions.json`` file.
    """
    file = tmp_path / "functions.json"
    file.write_text(json.dumps(sample_functions_def, indent=2))
    return file


@pytest.fixture
def make_input_file(tmp_path: Path) -> Callable[[str], Path]:
    """Return a function to create an ``input.json`` file.

    The created file has the exact format expected by ``--input``:
    ``[{"prompt": "..."}]``.  Call the function with the desired prompt
    to test targeted end to end scenarios.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        A function that takes a prompt string and returns the path to the
        created JSON file.
    """
    def _create(prompt: str) -> Path:
        file = tmp_path / "input.json"
        file.write_text(json.dumps([{"prompt": prompt}]))
        return file
    return _create


@pytest.fixture
def vocab_file(tmp_path: Path, vocab_small: Dict[int, str]) -> Path:
    """Temporary JSON file containing the test vocabulary.

    ``json.dumps`` automatically converts the integer keys of
    ``vocab_small`` to strings (``"0"``, ``"1"``, …).  This is the
    "Case A" format handled by ``load_vocab``, so the file can
    be used directly in loading tests.

    Args:
        tmp_path: Temporary directory.
        vocab_small: Reduced vocabulary fixture.

    Returns:
        Path to the ``vocab.json`` file.
    """
    file = tmp_path / "vocab.json"
    file.write_text(json.dumps(vocab_small))
    return file


def _group_stats(terminalreporter: TerminalReporter) -> dict[str, TestCounts]:
    """Calcola statistiche per modulo di test."""
    groups = {
        "test_tokenizer.py":            "tokenizer unit tests",
        "test_constrained_decoder.py":  "constrained decoder unit tests",
        "test_main.py":                 "main integration tests",
        "test_cli.py":                  "CLI end‑to‑end tests",
    }
    stats: dict[str, TestCounts] = {
        label: TestCounts(passed=0, failed=0) for label in groups.values()
    }

    for rep in terminalreporter.stats.get('passed', []):
        mod = rep.nodeid.split("::")[0].split("/")[-1]
        label = groups.get(mod)
        if label:
            stats[label]["passed"] += 1

    for rep in terminalreporter.stats.get('failed', []):
        mod = rep.nodeid.split("::")[0].split("/")[-1]
        label = groups.get(mod)
        if label:
            stats[label]["failed"] += 1

    return stats


def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
    exitstatus: int,
    config: Config,
) -> None:
    """Print a customized summary of the test suite results.

    It displays the total number of tests run, the number of failed tests,
    and a breakdown by group, using formatting adapted to the width of the
    terminal.
    """
    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    total = passed + failed

    if exitstatus == 0:
        status_line = f"All {total} tests passed"
    else:
        status_line = f"{failed} test(s) failed out of {total}"

    # Usa la larghezza del terminale, con un minimo di 60 colonne
    try:
        W = terminalreporter._tw.fullwidth
    except AttributeError:
        W = 80
    W = max(W, 60)
    sep = "=" * W

    def box_line(text: str) -> str:
        return f"║ {text.center(W - 4)} ║"

    terminalreporter.line(sep)
    terminalreporter.line(box_line("TEST SUITE SUMMARY"))
    terminalreporter.line(sep)
    terminalreporter.line(box_line(status_line))
    terminalreporter.line(sep)

    terminalreporter.line(" Groups:")
    group_data = _group_stats(terminalreporter)
    for label, counts in group_data.items():
        p, f = counts["passed"], counts["failed"]
        total_g = p + f
        if total_g == 0:
            continue
        if f == 0:
            mark = "✓"
        else:
            mark = "✗"
        # Allineamento dinamico: 3 spazi di indentazione + mark + label
        msg = f"   {mark} {label}  ({p} passed, {f} failed)"
        terminalreporter.line(msg)

    terminalreporter.line("")
    terminalreporter.line(sep)
