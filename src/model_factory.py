"""Model factory: builds concrete model instances."""
from __future__ import annotations

from typing import Optional

from .model_protocol import ModelProtocol
from .model_adapter import SmallLLMAdapter

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


def build_model(model_name: str) -> ModelProtocol:
    """Create and return a concrete model based on the given name.

    Args:
        model_name: Model identifier (``"qwen06b"``, ``"Qwen25"``).

    Returns:
        An instance of ``ModelProtocol`` ready to use.

    Raises:
        RuntimeError: if the real model requires ``llm_sdk`` but it is
            not installed.
        ValueError: if the model name is not recognised.
    """
    if model_name in ("qwen06b", "Qwen25"):
        if Small_LLM_Model is None:
            raise RuntimeError(
                "llm_sdk required but not installed. "
                "Run 'make install' and ensure "
                "llm_sdk/ is present."
            )
        if model_name == "qwen06b":
            return SmallLLMAdapter(Small_LLM_Model(), model_name="qwen06b")
        elif model_name == "Qwen25":
            return SmallLLMAdapter(
                Small_LLM_Model("Qwen/Qwen2.5-1.5B-Instruct"),
                model_name="Qwen25"
            )
    raise ValueError(f"Unsupported model: {model_name}")
