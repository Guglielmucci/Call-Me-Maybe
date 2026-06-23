"""Utilities for vocabulary loading and fast search structures.

Provides:
- load_vocab: read and interpret JSON vocabulary files.
- TokenTrie: trie structure for token sequences, used by the decoder
  for incremental validation of names and values.
"""

import json
from typing import Set


def load_vocab(vocab_path: str) -> dict[int, str]:
    """Load vocabulary from a JSON file and return an id -> token mapping.

    The expected format can be:
    - A dictionary where keys are numeric IDs and values are tokens.
    - A dictionary mapping token -> id (integer values).
    - A Hugging Face format with a 'model' key containing 'vocab'.

    Args:
        vocab_path: Path to the vocabulary JSON file.

    Returns:
        Dictionary mapping integer IDs to token strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the format is not recognised.
    """
    with open(vocab_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Case A: numeric keys (ID -> token)
        if all(isinstance(k, str) and k.isdigit() for k in data.keys()):
            return {int(k): v for k, v in data.items()}
        # Case B: token -> id (integer values)
        if all(isinstance(v, int) for v in data.values()):
            return {v: k for k, v in data.items()}
    # Case C: Hugging Face format
    if isinstance(data, dict) and "model" in data and "vocab" in data["model"]:
        vocab = data["model"]["vocab"]
        if isinstance(vocab, dict):
            return {v: k for k, v in vocab.items()}
    raise ValueError(
        "Unrecognised vocabulary format. "
        "Expected: ID->token mapping, token->ID mapping, "
        "or Hugging Face format."
    )


class TokenTrie:
    """Trie for token sequences, used to validate function names and booleans.

    Each node represents a token; an ``is_end`` node marks the end of a
    valid sequence. Allows retrieval of the next allowed tokens given a
    partial sequence.

    Attributes:
        children: Dict mapping a token id to the corresponding child node.
        is_end: True if the node represents the end of a complete sequence.
    """

    #  TokenTrie --------------------------------------------------------------
    # "Naviga TOKEN ID, non caratteri. Serve al decoder per sapere,
    #  dato un pezzo di sequenza già generata (es. i primi 2 token del
    #  nome di una funzione), quali token id possono seguire per restare
    #  dentro un nome di funzione valido o un booleano valido (true/false).

    def __init__(self) -> None:
        """Initialise an empty trie."""
        self.children: dict[int, "TokenTrie"] = {}
        self.is_end: bool = False

    def insert(self, token_ids: list[int]) -> None:
        """Insert a sequence of token ids into the trie.

        Args:
            token_ids: List of token ids that make up the sequence.

        Raises:
            ValueError: If the sequence is empty.
        """
        if not token_ids:
            raise ValueError("Cannot insert an empty sequence.")
        node: "TokenTrie" = self
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TokenTrie()
            node = node.children[tid]
        node.is_end = True

    def next_tokens(self, prefix: list[int]) -> Set[int]:
        """Return the set of token ids that can follow a given prefix.

        Args:
            prefix: Sequence of token ids already generated.

        Returns:
            Set of token ids valid as the next step.
        """
        node: "TokenTrie" = self
        for tid in prefix:
            if tid not in node.children:
                return set()
            node = node.children[tid]
        return set(node.children.keys())

    def is_complete(self, token_ids: list[int]) -> bool:
        """Check whether the token sequence corresponds to a complete path.

        Args:
            token_ids: Sequence of token ids to verify.

        Returns:
            True if the sequence ends at an ``is_end`` node, False otherwise.
        """
        node: "TokenTrie" = self
        for tid in token_ids:
            if tid not in node.children:
                return False
            node = node.children[tid]
        return node.is_end
