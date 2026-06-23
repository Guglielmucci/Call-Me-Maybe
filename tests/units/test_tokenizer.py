"""Unit tests for the custom tokenizer.

The tokenizer reproduces the behaviour of the SentencePiece model used
by the LLM:
- Spaces are converted to the special character 'Ġ' (U+0120) to match
  the way the LLM represents leading spaces.
- Newlines become 'Ċ' (U+010A).
- Encoding uses greedy longest match against a fixed vocabulary.
- Decoding reverses the process, restoring original whitespace.

These properties are critical because the constrained decoder must
map vocabulary tokens back to their string representations to decide
which tokens are valid at each generation step. The tests here
establish the correctness of this mapping, which is then used in
``test_constrained_decoder.py`` to verify the whole pipeline.
"""

import pytest
from src.tokenizer import CustomTokenizer


@pytest.fixture
def tiny_vocab() -> dict[int, str]:
    """Create minimal vocabulary that still exercises all tokenizer."""
    return {
        0: "a",
        1: "b",
        2: "ab",
        3: "Ġ",
        4: "Ċ",
        5: "hello",
        6: "world",
        7: " ",
        8: "true",
        9: "false",
        10: "123",
    }


@pytest.fixture
def tok(tiny_vocab: dict[int, str]) -> CustomTokenizer:
    """Pre‑built tokenizer from the tiny vocabulary."""
    return CustomTokenizer(tiny_vocab)


class TestEncoding:
    """Tests for the ``encode`` method."""

    def test_single_known_token(self, tok: CustomTokenizer) -> None:
        """A single token from the vocabulary is encoded to its ID."""
        assert tok.encode("a") == [0]

    def test_longest_match_priority(self, tok: CustomTokenizer) -> None:
        """The encoder chooses the longest possible match.

        Since both ``"a"`` and ``"ab"`` are in the vocabulary, ``"ab"``
        must be one token (ID 2), not two ``"a"`` tokens.
        """
        assert tok.encode("ab") == [2]
        assert tok.encode("aba") == [2, 0]      # "ab" + "a"

    def test_space_becomes_gspecial(self, tok: CustomTokenizer) -> None:
        """A plain space is converted to the special token ``Ġ``.

        This mimics the way the real LLM tokenizer attaches a leading
        space to the following token. The decoder later uses this
        transformation to correctly match strings that span multiple
        tokens.
        """
        assert tok.encode(" hello") == [3, 5]   # Ġ + hello

    def test_newline_becomes_cspecial(self, tok: CustomTokenizer) -> None:
        """A newline is converted to ``Ċ``."""
        assert tok.encode("a\nb") == [0, 4, 1]  # a + Ċ + b

    def test_unknown_character_uses_fallback(
            self, tok: CustomTokenizer
    ) -> None:
        """Characters not in the vocabulary are replaced by an UNK token.

        This prevents crashes during generation, though the constrained
        decoder should never allow such tokens in valid JSON.
        """
        result = tok.encode("x")   # 'x' not in tiny_vocab
        assert len(result) == 1
        assert result[0] == 0      # default UNK fallback

    def test_empty_string(self, tok: CustomTokenizer) -> None:
        """An empty string produces an empty list of token IDs."""
        assert tok.encode("") == []


class TestDecoding:
    """Tests for the ``decode`` method."""

    def test_simple_decode(self, tok: CustomTokenizer) -> None:
        """A list of token IDs is converted back to the original string."""
        assert tok.decode([5, 3, 6]) == "hello world"

    def test_gspecial_restores_space(self, tok: CustomTokenizer) -> None:
        """The special token ``Ġ`` is rendered as a normal space.

        This is the inverse of the encoding step and guarantees that
        the final output is human‑readable JSON, not a string full of
        ``Ġ`` symbols.
        """
        assert tok.decode([3, 5]) == " hello"

    def test_cspecial_restores_newline(self, tok: CustomTokenizer) -> None:
        """``Ċ`` is restored to a newline character."""
        assert tok.decode([0, 4, 1]) == "a\nb"

    def test_unknown_id_is_ignored(self, tok: CustomTokenizer) -> None:
        """Controll if a token ID is not in the vocabulary.

        Decode produces an empty string for that position, keeping the rest.
        """
        assert tok.decode([5, 99, 6]) == "helloworld"

    def test_decode_empty_list(self, tok: CustomTokenizer) -> None:
        """Decoding an empty list returns an empty string."""
        assert tok.decode([]) == ""


class TestInitialization:
    """Tests for creating the tokenizer from different sources."""

    def test_from_dict(self, tiny_vocab: dict[int, str]) -> None:
        """The tokenizer can be initialised with a plain dictionary."""
        tok = CustomTokenizer(tiny_vocab)
        assert tok.encode("a") == [0]

    def test_from_file(self, vocab_file: str) -> None:
        """The tokenizer can load its vocabulary from a JSON file.

        ``vocab_file`` is a temporary fixture containing the test
        vocabulary in the format returned by
        ``model.get_path_to_vocabulary_json()``.
        """
        tok = CustomTokenizer(vocab_file)
        # Sanity check: one token from the standard test vocabulary
        result = tok.encode("fn_add_numbers")
        assert result  # list of token IDs
