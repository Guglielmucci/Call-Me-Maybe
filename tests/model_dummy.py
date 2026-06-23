"""Create a Dummy Model for testing."""

import tempfile
import json
from pathlib import Path
from typing import Optional

from src.model_protocol import ModelProtocol


class DummyLLM(ModelProtocol):
    """Return a random logits and a vocabulary file.

    The vocabulary is a dict mapping token IDs to strings. The logits
    array length equals max(token ID) + 1, not the number of entries.
    """

    def __init__(self, vocab: Optional[dict[int, str]] = None) -> None:
        """Initialize the dummy model with an optional vocabulary.

        If no vocabulary is provided, a default one is created containing
        typical JSON tokens and some example functions.
        A temporary file is also created that exposes the vocabulary
        in JSON format, simulating the behavior of real models.
        """
        if vocab is None:
            vocab = {
                0: "{", 1: "}", 2: '"', 3: ":", 4: ",",
                5: '"name"', 6: '"parameters"',
                7: "fn_add", 8: "fn_greet", 9: "fn_reverse_string",
                10: "a", 11: "b", 12: "name", 13: "s",
                14: "true", 15: "false", 16: "1", 17: "2", 18: "3",
                19: "hello", 20: "world",
            }
        self._vocab = vocab
        # Create a temporary file that simulates the JSON vocabulary file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        self._vocab_path = Path(tmp.name)
        json.dump({str(k): v for k, v in self._vocab.items()}, tmp)
        tmp.close()

    def __del__(self) -> None:
        """Remove the temporary file when the object is destroyed."""
        if hasattr(self, "_vocab_path") and self._vocab_path.exists():
            try:
                self._vocab_path.unlink()
            except Exception:
                pass

    def set_vocab(self, vocab: dict[int, str]) -> None:
        """Replace the vocabulary and regenerates the temporary file."""
        self._vocab = vocab
        with open(self._vocab_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self._vocab.items()}, f)

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the JSON vocabulary file."""
        return str(self._vocab_path)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return a deterministic logit vector for the given inputs."""
        size = max(self._vocab.keys()) + 1 if self._vocab else 1
        return [0.0] * size
