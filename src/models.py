"""Pydantic models for validating function definitions."""

from pathlib import Path
from typing import Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class ParameterDef(BaseModel):
    """Typed definition of a function parameter.

    Attributes:
        type: Allowed type for the parameter:
        ``"number"``, ``"string"``, ``"boolean"``.
        format: Optional parameter format hint for specialized string handling,
        e.g. ``"regex"`` or ``"replacement"``.
        subtype: Optional alias for additional parameter type metadata.
    """

    type: str
    format: Optional[str] = None
    subtype: Optional[str] = None
    allowed_values: Optional[list[str]] = None


class FunctionDef(BaseModel):
    """Complete definition of a callable function.

    Attributes:
        name: Function name.
        description: Textual description of what the function does.
        parameters: Dictionary mapping parameter names to their definitions.
        returns: Def. of the return type (contains only the ``type`` field).
    """

    name: str
    description: str
    parameters: dict[str, ParameterDef]
    returns: ParameterDef


class TestItem(BaseModel):
    """Single prompt to process.

    Attributes:
        prompt: Natural language request.
    """

    prompt: str


class OutputItem(BaseModel):
    """Output element representing a resolved function call.

    Attributes:
        prompt: Original prompt.
        name: Selected function name.
        parameters: Concrete arguments with valid types.
    """

    prompt: str
    name: str
    parameters: dict[str, Union[int, float, str, bool, None]]
    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseModel):
    """Globally validated application configuration.

    Attributes:
        functions_definition: Path to the JSON file with function definitions.
        input: Path to the JSON file with test prompts.
        output: Path where to write the JSON results.
        model: LLM model identifier (default: ``"qwen06b"``).
        verbose: If ``True``, enables detailed debug logs.
        timeout: Timeout in seconds per prompt (0 = no limit).
    """

    functions_definition: Path = Field(
        default=Path("data/input/functions_definition.json"),
        description="Path to the JSON file with function definitions.",
    )
    input: Path = Field(
        default=Path("data/input/function_calling_tests.json"),
        description="Path to the JSON file with prompts to process.",
    )
    output: Path = Field(
        default=Path("data/output/function_calling_results.json"),
        description="Path where to write the output JSON file.",
    )
    model: str = Field(
        default="qwen06b",
        description="LLM model to use (default: qwen06b).",
    )
    verbose: bool = Field(
        default=False,
        description="If True, shows detailed debug logs.",
    )
    timeout: float = Field(
        default=30.0,
        description="Timeout in seconds per generation (0 = no limit).",
    )

    model_config = ConfigDict(extra="forbid")
