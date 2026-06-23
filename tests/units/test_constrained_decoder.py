"""Tests for the constrained JSON decoder (JSONStateManager).

These tests verify that the automaton correctly restricts which tokens
are allowed at each generation step. They use the custom tokenizer
(tested in ``test_tokenizer.py``) to encode structural elements and
function names, demonstrating the complete integration between
tokenization and constrained decoding.
"""

import logging
from typing import Dict, List, Set

import pytest

from src.schema import JSONStateManager, ParseState
from src.token_utils import TokenTrie
from src.models import FunctionDef
from src.tokenizer import CustomTokenizer


@pytest.fixture
def fixed_tokens(vocab_small: Dict[int, str]) -> Dict[str, int]:
    """Symbolic names for the JSON structural tokens.

    The values are token IDs taken from the shared ``vocab_small``.
    """
    return {
        "open_brace": 0,
        "close_brace": 1,
        "quote": 2,
        "colon": 3,
        "comma": 4,
        "name_key": 5,
        "params_key": 6,
        "open_square": 25,
        "close_square": 26,
        "star": 27,
    }


@pytest.fixture
def func_trie(
    vocab_small: Dict[int, str], mock_tokenizer: CustomTokenizer
) -> TokenTrie:
    """Trie containing the token sequences of all sample function names.

    The tokenizer is used to encode each function name (e.g.
    ``"fn_add_numbers"``), ensuring the trie uses the same token IDs
    that the decoder will see.
    """
    trie = TokenTrie()
    for name in ("fn_add_numbers", "fn_greet", "fn_reverse_string"):
        trie.insert(mock_tokenizer.encode(name))
    return trie


@pytest.fixture
def bool_trie(
    vocab_small: Dict[int, str], mock_tokenizer: CustomTokenizer
) -> TokenTrie:
    """Trie for the boolean values ``"true"`` and ``"false"``."""
    trie = TokenTrie()
    trie.insert(mock_tokenizer.encode("true"))
    trie.insert(mock_tokenizer.encode("false"))
    return trie


@pytest.fixture
def param_key_ids(
    sample_funcs: List[FunctionDef], mock_tokenizer: CustomTokenizer
) -> Dict[str, Dict[str, List[int]]]:
    """For each function, a mapping from parameter name to its enc JSON key.

    For example ``"a"`` becomes ``[2, 15, 2]`` (quote, token for 'a', quote).
    """
    result: Dict[str, Dict[str, List[int]]] = {}
    for func in sample_funcs:
        key_map: Dict[str, List[int]] = {}
        for pname in func.parameters:
            key_map[pname] = mock_tokenizer.encode(f'"{pname}"')
        result[func.name] = key_map
    return result


@pytest.fixture
def string_safe_tokens(vocab_small: Dict[int, str]) -> Set[int]:
    """Tokens that are allowed inside a JSON string.

    They must not contain the double quote character, nor the special
    whitespace tokens ``Ġ`` and ``Ċ``.
    """
    return {
        tid
        for tid, tok in vocab_small.items()
        if tok and '"' not in tok and tok not in ("Ġ", "Ċ")
    }


@pytest.fixture
def state_mgr(
    func_trie: TokenTrie,
    sample_funcs: List[FunctionDef],
    fixed_tokens: Dict[str, int],
    param_key_ids: Dict[str, Dict[str, List[int]]],
    bool_trie: TokenTrie,
    string_safe_tokens: Set[int],
    vocab_small: Dict[int, str],
) -> JSONStateManager:
    """Use the real tokenizer in a Configured JSONStateManager .

    This fixture is the heart of the integration tests: it wires
    together the tokenizer‑produced tries and key sequences with the
    automaton, exactly as in the production code.
    """
    func_defs = {f.name: f for f in sample_funcs}
    mgr = JSONStateManager(
        func_trie=func_trie,
        func_defs=func_defs,
        fixed_tokens=fixed_tokens,
        param_key_ids=param_key_ids,
        bool_trie=bool_trie,
        string_safe_tokens=string_safe_tokens,
        name_key_ids=[fixed_tokens["name_key"]],
        params_key_ids=[fixed_tokens["params_key"]],
        enum_tries={},
    )
    mgr.set_vocab(vocab_small)
    return mgr


class TestCompleteGeneration:
    """End to end sequences that exercise every state of the automaton.

    Each test feeds a predetermined list of token IDs, simulating the
    output of the constrained decoder, and then checks the final parsed
    result. This demonstrates that the automaton correctly guides the
    generation from opening brace to the ``DONE`` state.
    """

    def test_generate_add_numbers(self, state_mgr: JSONStateManager) -> None:
        """Generate Function for add numbers.

        The sequence uses the exact token IDs from ``vocab_small``:
        - 0: ``{``
        - 5: ``"name"`` (because in this vocabulary the whole key is one token)
        - 3: ``:``
        - 2: ``"``
        - 12: ``fn_add_numbers``
        - 2: ``"``
        - ... etc.
        """
        seq = [
            0, 5, 3, 2, 12, 2, 4,           # {"name":"fn_add_numbers",
            6, 3, 0,                        # "parameters":{
            2, 15, 2, 3, 20, 4,             # "a":42,
            2, 16, 2, 3, 22,                # "b":3.14
            1, 1                            # }}
        ]
        for tid in seq:
            allowed = state_mgr.allowed_tokens()
            assert tid in allowed, (
                f"Token {tid} ({state_mgr.vocab.get(tid, '?')}) "
                f"not allowed in state {state_mgr.state.name}"
            )
            state_mgr.advance(tid)

        assert state_mgr.state == ParseState.DONE
        result = state_mgr.get_result()
        assert result["name"] == "fn_add_numbers"
        assert result["parameters"] == {"a": 42.0, "b": 3.14}

    def test_generate_greet(self, state_mgr: JSONStateManager) -> None:
        """Generate ``{"name":"fn_greet","parameters":{"name":"John"}}``.

        Note: in ``vocab_small`` the parameter key ``"name"`` is a single
        token (5), so we don't need separate quote tokens for it.
        """
        seq = [
            0, 5, 3, 2, 13, 2, 4,               # {"name":"fn_greet",
            6, 3, 0,                            # "parameters":{
            5, 3, 2, 21, 2,                     # "name":"John"
            1, 1                                # }}
        ]
        for tid in seq:
            allowed = state_mgr.allowed_tokens()
            assert tid in allowed
            state_mgr.advance(tid)

        assert state_mgr.state == ParseState.DONE
        result = state_mgr.get_result()
        assert result["name"] == "fn_greet"
        assert result["parameters"] == {"name": "John"}


class TestValueTypeConstraints:
    """Tests verify the automaton restricts values to the declared type."""

    def test_boolean_allows_only_true_false(
        self, state_mgr: JSONStateManager
    ) -> None:
        """When the parameter type is ``boolean``.

        In the test vocabulary, the whole words are single tokens (18 and 19),
        so the automaton returns them directly without a preceding quote.
        """
        # Force the automaton into VALUE_BOOLEAN state
        state_mgr.state = ParseState.VALUE_BOOLEAN
        state_mgr.current_func = state_mgr.func_defs["fn_add_numbers"]
        state_mgr.param_list = ["a", "b"]
        state_mgr.param_idx = 0
        state_mgr.current_param_type = "boolean"
        state_mgr.value_tokens = []

        allowed = state_mgr.allowed_tokens()
        # The only valid starting tokens are the boolean literals
        assert 18 in allowed                            # "true"
        assert 19 in allowed                            # "false"

        # The quote token should NOT be allowed at this point
        assert state_mgr.fixed["quote"] not in allowed

        # Consume the "true" token
        state_mgr.advance(18)
        allowed_after = state_mgr.allowed_tokens()

        # After a complete boolean, we can close the value:
        # comma/close_brace
        assert state_mgr.fixed["close_brace"] in allowed_after
        if state_mgr.param_idx + 1 < len(state_mgr.param_list):
            assert state_mgr.fixed["comma"] in allowed_after

    def test_string_safe_tokens_are_allowed(
        self, state_mgr: JSONStateManager
    ) -> None:
        """Control Inside a string value.

        The automaton uses a pre_computed ``string_safe_tokens`` set.
        """
        state_mgr.state = ParseState.VALUE_STRING
        state_mgr.current_param_name = "some_param"
        state_mgr.value_tokens = []

        allowed = state_mgr.allowed_tokens()
        # The token for "a" (15) must be permitted
        assert 15 in allowed
        # The closing quote is always allowed
        assert state_mgr.fixed["quote"] in allowed

    def test_string_length_limit_enforced(
        self, state_mgr: JSONStateManager
    ) -> None:
        """Controll after reaching the maximum number of tokens for a string.

        This prevents infinite generation and ensures the JSON can always
        be closed.
        """
        state_mgr.state = ParseState.VALUE_STRING
        # Fill the buffer up to the limit (default 30)
        state_mgr.value_tokens = [15] * 100

        allowed = state_mgr.allowed_tokens()
        assert allowed == {state_mgr.fixed["quote"]}


class TestErrorHandling:
    """Tests for degradation when the model output is unexpected."""

    def test_unknown_function_falls_back_to_first(
        self, state_mgr: JSONStateManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Produces an unknown function name.

        The automaton logs an error and falls back to the first available
        function. This ensures that a partially correct generation does not
        crash the entire pipeline.
        """
        caplog.set_level(logging.ERROR)
        del state_mgr.func_defs["fn_greet"]

        # Simulate having just decoded the name "fn_greet" -----------------
        state_mgr.state = ParseState.EXPECT_FUNC_NAME
        state_mgr.func_name_tokens = [13]            # token for "fn_greet"
        state_mgr.advance(state_mgr.fixed["quote"])  # closing quote

        assert "Unknown function name decoded: 'fn_greet'" in caplog.text
        assert state_mgr.current_func is not None
        assert state_mgr.current_func.name == "fn_add_numbers"

    def test_missing_parameter_value_warning(
        self, state_mgr: JSONStateManager, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Controll When a parameter has no value.

        Warning is logged and the value is set to ``None`` in the final output.
        This allows the pipeline to produce a result even if the model
        stopped prematurely.
        """
        caplog.set_level(logging.WARNING)
        state_mgr.current_func = state_mgr.func_defs["fn_add_numbers"]
        state_mgr.param_list = ["a", "b"]
        state_mgr.param_values = [42.0]
        state_mgr.state = ParseState.DONE

        result = state_mgr.get_result()
        assert "Missing value for parameter 'b'" in caplog.text
        assert result["parameters"]["b"] is None

    def test_incomplete_generation_adds_error_field(
        self, state_mgr: JSONStateManager
    ) -> None:
        """Controll if the automaton is not in the ``DONE`` state.

        ``get_result`` includes an ``_error`` field explaining the situation.
        This aids debugging and prevents silent failures.
        """
        state_mgr.current_func = state_mgr.func_defs["fn_add_numbers"]
        state_mgr.param_list = ["a", "b"]
        state_mgr.param_values = [42.0]
        # state is still not DONE
        result = state_mgr.get_result()
        assert "_error" in result
        assert "Incomplete generation" in result["_error"]

    def test_force_close_string_produces_valid_value(
        self, state_mgr: JSONStateManager
    ) -> None:
        """Calling ``force_close_current_value`` on a string value.

        Correctly finishes the parameter and moves to the next expected
        token (comma or closing brace).
        """
        state_mgr.state = ParseState.VALUE_STRING
        state_mgr.current_param_type = "string"
        state_mgr.value_tokens = [21]
        state_mgr.param_list = ["name"]
        state_mgr.param_idx = 0
        state_mgr.current_func = state_mgr.func_defs["fn_greet"]

        state_mgr.force_close_current_value()
        assert state_mgr.state == ParseState.EXPECT_PARAM_COMMA_OR_CLOSE
        assert state_mgr.param_values == ["John"]
