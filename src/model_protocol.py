"""Public protocol for interacting with LLM models.

Defines the minimum interface required by the constrained decoding
project.  Different model implementations (real SDK models, test
dummies, etc.) can be used interchangeably without accessing private
SDK internals.
"""

from typing import Any, Protocol


class ModelProtocol(Protocol):
    """Public interface that every LLM model adapter must satisfy.

    The methods declared here are the only ones called by the
    :class:`ConstrainedDecoder` and the main module:

    * :meth:`get_path_to_vocab_file` – returns the vocabulary file path.
    * :meth:`get_logits_from_input_ids` – returns logits for a prefix.
    * :meth:`set_vocab` – replaces the vocabulary (used by dummy models).

    Any class that provides these three methods implicitly implements
    the protocol, enabling static type-checking with ``mypy`` without
    a concrete base class.
    """

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the vocabulary JSON file.

        Returns:
            The filesystem path to the vocabulary file as a string.
        """
        ...

    def get_logits_from_input_ids(self, input_ids: Any) -> Any:
        """Return the model's logits for the given input prefix.

        Args:
            input_ids: The tokenized prefix (format depends on the
                concrete model, typically a list of integers).

        Returns:
            The raw logits, as an array-like object that can be
            converted to a NumPy array for further processing.
        """
        ...

    def set_vocab(self, vocab: dict[int, str]) -> None:
        """Replace the internal vocabulary mapping.

        This is primarily used by dummy models for testing; real SDK
        models may ignore it.

        Args:
            vocab: A mapping from token ID to string representation.
        """
        ...
