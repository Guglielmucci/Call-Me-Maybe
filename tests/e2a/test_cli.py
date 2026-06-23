"""End to end tests for the function calling CLI.

These tests simulate a real invocation of the program, but run entirely
in process to avoid external dependencies.  A custom ``DummyLLM`` model
(in ``tests/model_dummy.py``) returns deterministic logits, and we mock
``build_model`` to inject it.  This allows us to verify the complete
pipeline — argument parsing, file loading, constrained decoding, and
JSON output — without relying on the real SDK or hardware.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from src.__main__ import main
from tests.model_dummy import DummyLLM


def test_cli_dummy_model_output_structure(tmp_path: Path) -> None:
    """Verify that the CLI produces a formed output when using a dummy model.

    **Scenario**
    - A single function definition (``fn_test`` with one numeric
      parameter ``x``) is provided.
    - A single prompt ``"Return 42"`` is fed to the system.
    - The model factory is mocked to return a ``DummyLLM``, which
      produces deterministic (all zero) logits.
    - We call ``main()`` as if the CLI were invoked with these files.

    **Expected behaviour**
    - The program runs without errors (no ``SystemExit``).
    - An output file is created and contains a JSON array with one
      element.
    - That element has the keys ``name`` and ``parameters``, the
      function name is ``"fn_test"`` (the only possible choice), and
      the parameter ``x`` is a number (type matches the definition).

    **Why in process?**
    Running the CLI via ``subprocess`` would require the real
    ``build_model`` to recognise the ``"dummy"`` identifier.  By
    patching ``build_model`` inside ``src.__main__`` and calling
    ``main()`` directly, we keep the dummy model confined to the test
    suite while still exercising the entire application logic.
    """
    # ========================================================================
    #   Arrange
    # ========================================================================
    funcs = [
        {
            "name": "fn_test",
            "description": "A test function",
            "parameters": {"x": {"type": "number"}},
            "returns": {"type": "number"},
        }
    ]
    prompts = [{"prompt": "Return 42"}]

    func_file = tmp_path / "functions.json"
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    func_file.write_text(json.dumps(funcs))
    input_file.write_text(json.dumps(prompts))

    test_args = [
        "prog",
        "--model", "dummy",
        "--functions_definition", str(func_file),
        "--input", str(input_file),
        "--output", str(output_file),
    ]

    dummy_instance = DummyLLM()
    # ========================================================================
    #   Act
    # ========================================================================
    # Patch build_model inside src.__main__ to inject the dummy model
    with patch("src.__main__.build_model", return_value=dummy_instance):
        with patch.object(sys, "argv", test_args):
            main()

    # ========================================================================
    #   Assert
    # ========================================================================
    assert output_file.exists(), "Output file was not created"
    data = json.loads(output_file.read_text())
    assert isinstance(data, list), "Top-level output must be a JSON array"
    assert len(data) == 1, (
        "There should be exactly one result for the single prompt"
    )
    result = data[0]
    assert result["name"] == "fn_test", (
        "With only one function defined, the decoder must choose 'fn_test'"
    )
    assert isinstance(result["parameters"]["x"], (int, float)), (
        "Parameter 'x' must be a number as declared"
    )
