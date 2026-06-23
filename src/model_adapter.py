"""Adapter that makes the concrete model compatible with ModelProtocol."""

from typing import Any
from .model_protocol import ModelProtocol


class SmallLLMAdapter(ModelProtocol):
    """Wrapper around ``Small_LLM_Model`` that exposes the expected interface.

    Implements :class:`ModelProtocol`, ensuring that static type checkers
    like ``mypy`` recognise the required methods without modifying the
    original SDK.  The adapter works with any model backend that provides
    ``Small_LLM_Model`` (e.g. Qwen2.5-1.5B-Instruct, Qwen0.6B).

    Attributes:
        _model: The concrete model instance.
        _model_name: Lowercase model identifier (``"qwen06b"`` or
            ``"Qwen25"``) used for logging and to select the correct
            vocabulary file.
    """

    def __init__(self, model: Any, model_name: str = "") -> None:
        """Initialise the adapter.

        Args:
            model: An instance of ``Small_LLM_Model`` (or a compatible
                object) providing ``get_logits_from_input_ids`` and
                vocabulary file paths.
            model_name: Model identifier such as ``"qwen06b"`` or
                ``"Qwen25"``.  Only used for logging; the actual model
                behaviour is determined by the ``model`` argument.
        """
        self._model = model
        self._model_name = model_name

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the vocabulary JSON file for the tokenizer.

        All supported models (Qwen2.5, Qwen0.6B) use a standard JSON
        vocabulary file, so this method simply forwards the call to the
        underlying model.

        Returns:
            The filesystem path to the vocabulary file as a string.
        """
        return str(self._model.get_path_to_vocab_file())

    def get_logits_from_input_ids(self, input_ids: Any) -> Any:
        """Forward the logits request to the concrete model.

        Args:
            input_ids: Tokenised input sequence (typically a list of
                integers or a tensor).

        Returns:
            Raw logits as returned by the model's forward pass.
        """
        return self._model.get_logits_from_input_ids(input_ids)

    def set_vocab(self, vocab: dict[int, str]) -> None:
        """No‑op – the SDK already manages the full vocabulary.

        The method exists only to satisfy :class:`ModelProtocol`.

        Args:
            vocab: Token‑ID to string mapping (ignored).
        """
        pass
