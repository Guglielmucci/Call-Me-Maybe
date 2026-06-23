"""State automaton for constrained JSON generation."""

from __future__ import annotations
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from .models import FunctionDef
from .token_utils import TokenTrie
import re

logger = logging.getLogger(__name__)


class ParseState(Enum):
    """States of the JSON generation automaton.

    The automaton walks a fixed path through the JSON structure:
    opening brace → ``"name"`` key → function name → ``"parameters"`` key
    → parameter key/value pairs → closing braces → DONE.

    States prefixed with ``EXPECT_`` are deterministic (injected without
    model involvement). States ``VALUE_*`` require model sampling under
    token constraints.
    """

    EXPECT_OPEN_BRACE = 1
    EXPECT_NAME_KEY = 2
    EXPECT_NAME_COLON = 3
    EXPECT_FUNC_NAME_QUOTE = 4
    EXPECT_FUNC_NAME = 5
    EXPECT_COMMA_AFTER_NAME = 7
    EXPECT_PARAMS_KEY = 8
    EXPECT_PARAMS_COLON = 9
    EXPECT_PARAMS_OPEN = 10
    EXPECT_PARAM_KEY = 11
    EXPECT_PARAM_COLON = 12
    EXPECT_VALUE_START = 13
    EXPECT_VALUE_QUOTE = 20
    VALUE_NUMBER = 14
    VALUE_STRING = 15
    VALUE_BOOLEAN = 16
    EXPECT_PARAM_COMMA_OR_CLOSE = 17
    EXPECT_OBJECT_END = 18
    DONE = 19


class JSONStateManager:
    """State manager for constrained JSON generation.

    Tracks the current automaton state, accumulates value tokens, and
    exposes ``allowed_tokens()`` so the decoder can mask logits at each
    generation step. Structural tokens (braces, colons, commas, quotes,
    parameter keys) are injected deterministically by the decoder; only
    value tokens (numbers, strings, booleans) are sampled from the model.

    Attributes:
        func_trie: Trie of tokenized function names.
        func_defs: Dictionary of available function definitions.
        fixed: Mapping of structural token names to their token IDs
            (e.g. ``"open_brace"``, ``"quote"``).
        param_key_ids: Pre-encoded parameter key tokens per function.
        bool_trie: Trie for boolean values ``true`` and ``false``.
        string_safe_tokens: Set of token IDs safe for string value
            generation.
        max_string_tokens: Maximum tokens allowed in a string value.
        max_regex_tokens: Maximum tokens for a regex string value.
        max_replacement_tokens: Maximum tokens for a replacement string.
        enum_tries: Mapping ``(func_name, param_name) -> TokenTrie`` for
            allowed enum values.
        state: Current automaton state.
        vocab: Vocabulary mapping token IDs to strings.
        func_name_tokens: Accumulator for function name tokens.
        current_func: Currently selected function definition (or None).
        param_list: Ordered list of parameter names for current function.
        param_idx: Index of the parameter currently being processed.
        value_tokens: Accumulator for the current parameter value tokens.
        current_param_name: Name of the current parameter being filled.
        current_param_type: Type of the current parameter
            (``"string"``, ``"number"``, ``"integer"``, ``"boolean"``).
        param_values: List of parsed parameter values.
        param_formats: Optional semantic format constraints per parameter
            (e.g. ``"regex"``, ``"replacement"``).
    """

    def __init__(
        self,
        func_trie: TokenTrie,
        func_defs: Dict[str, FunctionDef],
        fixed_tokens: Dict[str, int],
        param_key_ids: Dict[str, Dict[str, List[int]]],
        bool_trie: TokenTrie,
        string_safe_tokens: Set[int],
        name_key_ids: List[int],
        params_key_ids: List[int],
        enum_tries: Dict[tuple[str, str], TokenTrie],
        max_string_tokens: int = 30,
        max_regex_tokens: int = 28,
        max_replacement_tokens: int = 5,
    ) -> None:
        """Initialize the JSON state manager.

        Args:
            func_trie: Trie for function name lookahead.
            func_defs: Available function definitions.
            fixed_tokens: Structural token IDs.
            param_key_ids: Pre-encoded parameter key tokens.
            bool_trie: Trie for boolean token matching.
            string_safe_tokens: Tokens allowed inside generic strings.
            name_key_ids: Pre-encoded ``"name"`` key tokens.
            params_key_ids: Pre-encoded ``"parameters"`` key tokens.
            enum_tries: Tries for enum value constraints.
            max_string_tokens: Token limit for string values.
            max_regex_tokens: Token limit for regex values.
            max_replacement_tokens: Token limit for replacement strings.
        """
        self.func_trie = func_trie
        self.func_defs = func_defs
        self.fixed = fixed_tokens
        self.param_key_ids = param_key_ids
        self.bool_trie = bool_trie
        self.string_safe_tokens = string_safe_tokens
        self.max_string_tokens = max_string_tokens
        self.max_regex_tokens = max_regex_tokens
        self.max_replacement_tokens = max_replacement_tokens
        self._in_escape: bool = False
        self.state = ParseState.EXPECT_OPEN_BRACE
        self.vocab: Dict[int, str] = {}
        self.func_name_tokens: List[int] = []
        self.current_func: Optional[FunctionDef] = None
        self.param_list: List[str] = []
        self.param_idx: int = 0
        self.value_tokens: List[int] = []
        self.current_param_name: Optional[str] = None
        self.current_param_type: Optional[str] = None
        self.param_values: List[Any] = []
        self._quoted_spans: Set[str] = set()
        self._all_candidates: Set[str] = set()
        self.name_key_tokens: List[int] = name_key_ids
        self.name_key_progress: int = 0
        self.params_key_tokens: List[int] = params_key_ids
        self.params_key_progress: int = 0
        self.param_key_progress: int = 0
        self.current_param_key_tokens: List[int] = []
        self._number_tokens: Set[int] = self._build_number_tokens()
        self._string_safe_tokens: Set[int] = set()
        self._cleaned_token_str: Dict[int, str] = {}
        self.enum_tries = enum_tries
        self._current_enum_trie: Optional[TokenTrie] = None
        self.param_formats: Dict[tuple[str, str], str] = {}
        for fname, fdef in func_defs.items():
            for pname, pdef in fdef.parameters.items():
                if pdef.format:
                    self.param_formats[(fname, pname)] = pdef.format
        self.current_param_format: Optional[str] = None
        self.regex_bracket_close_id: Optional[int] = None
        self.regex_paren_close_id: Optional[int] = None
        self._escape_token_ids: Set[int] = set()
        self.backslash_token_id: Optional[int] = None

    def set_vocab(self, vocab: Dict[int, str]) -> None:
        """Configure the vocabulary and rebuild dependent token sets.

        Args:
            vocab: Mapping from token ID to string representation.
        """
        self.vocab = vocab
        self._number_tokens = self._build_number_tokens()
        self._cleaned_token_str = {
            tid: tok.replace("Ġ", " ").replace("Ċ", " ")
            for tid, tok in vocab.items()
        }
        # Find token IDs for ] and ) ------------------------------------------
        for tid, tok in vocab.items():
            clean = tok.replace("Ġ", " ").replace("Ċ", " ")
            if clean == "]":
                self.regex_bracket_close_id = tid
            elif clean == ")":
                self.regex_paren_close_id = tid
        # Single backslash token ----------------------------------------------
        for tid, tok in vocab.items():
            clean = tok.replace("Ġ", " ").replace("Ċ", " ")
            if clean == "\\":
                self.backslash_token_id = tid
                break
        # Valid escape tokens ------------------------------------------------
        escape_chars = [
            '"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'
        ]
        self._escape_token_ids = set()
        for tid, tok in vocab.items():
            clean = self._cleaned_token_str[tid]
            if clean in escape_chars:
                self._escape_token_ids.add(tid)

    def set_prompt(self, prompt: str) -> None:
        """Extract candidate spans from the prompt for string anchoring.

        Args:
            prompt: The user's original prompt string.
        """
        quoted_spans: Set[str] = set()
        for m in re.finditer(r"'([^']+)'", prompt):
            quoted_spans.add(m.group(1))
        for m in re.finditer(r'"([^"]+)"', prompt):
            quoted_spans.add(m.group(1))
        all_candidates: Set[str] = set(quoted_spans)
        for word in prompt.split():
            w = word.strip("'\"(),.?!:;")
            if w:
                all_candidates.add(w)
        self._quoted_spans = quoted_spans
        self._all_candidates = all_candidates

    def _decode_partial_string(self) -> str:
        """Decode the current value buffer into a readable string.

        Returns:
            The partially decoded string, with special tokens replaced
            and basic escape handling applied.
        """
        if not self.value_tokens:
            return ""
        raw = "".join(self.vocab.get(tid, "") for tid in self.value_tokens)
        val = raw.replace("Ġ", " ").replace("Ċ", " ")
        val = val.replace("\\\\", "\\")
        val = val.replace('\\"', '"')
        return val

    def _build_number_tokens(self) -> Set[int]:
        """Identify all token IDs that consist solely of numeric characters.

        Returns:
            Set of token IDs valid for number / integer values.
        """
        if not self.vocab:
            return set()
        allowed = set("0123456789.-+eE")
        return {
            tid for tid, tok in self.vocab.items() if tok and all(
                c in allowed for c in tok
            )
        }

    def prepare_first_param(self) -> None:
        """Set up the token sequence for the first parameter key."""
        if self.current_func is None:
            return
        if self.param_idx < len(self.param_list):
            pname = self.param_list[self.param_idx]
            self.current_param_key_tokens = self.param_key_ids[
                self.current_func.name
            ][pname]
            self.param_key_progress = 0

    def get_current_param_key_tokens(self) -> List[int]:
        """Return the pre-encoded token sequence for the current param key.

        Returns:
            List of token IDs that form the current parameter's JSON key.
        """
        return self.current_param_key_tokens

    def transition_to_value_state(self) -> None:
        """Move from EXPECT_VALUE_START into the appropriate value state.

        Sets ``self.state`` based on the current parameter type.
        For strings also sets up the enum trie if available.
        """
        ptype = self.current_param_type
        if ptype in ("number", "integer"):
            self.state = ParseState.VALUE_NUMBER
        elif ptype == "string":
            if self.current_func and self.current_param_name:
                key = (self.current_func.name, self.current_param_name)
                self._current_enum_trie = self.enum_tries.get(key)
            else:
                self._current_enum_trie = None
            self.state = ParseState.EXPECT_VALUE_QUOTE
        elif ptype == "boolean":
            self.state = ParseState.VALUE_BOOLEAN
        else:
            self.state = ParseState.EXPECT_VALUE_QUOTE

    def _raw_value(self) -> str:
        """Return the raw concatenation of value tokens.

        Returns:
            The raw string representation of the accumulated value tokens.
        """
        return "".join(self.vocab.get(tid, "") for tid in self.value_tokens)

    def _cleaned_raw_value(self) -> str:
        """Return the raw value string with special markers replaced.

        Returns:
            A cleaned version of the raw value.
        """
        return self._raw_value().replace("Ġ", " ").replace("Ċ", " ")

    # ------------------------------------------------------------------
    # Core token filter: allowed_tokens
    # ------------------------------------------------------------------
    def allowed_tokens(self) -> Set[int]:
        """Compute the set of token IDs the model emit in the current state.

        This is the central constraint enforcer. It branches on the automaton
        state and returns the appropriate set of legal tokens, respecting
        function name trie, enum tries, boolean trie, string length limits,
        regex balancing, and prompt-based string anchoring.

        Returns:
            Set of allowed token IDs. An empty set signals a dead end.
        """
        state = self.state

        if state == ParseState.EXPECT_OPEN_BRACE:
            return {self.fixed["open_brace"]}
        if state == ParseState.EXPECT_NAME_KEY:
            if self.name_key_progress < len(self.name_key_tokens):
                return {self.name_key_tokens[self.name_key_progress]}
            return set()
        if state == ParseState.EXPECT_NAME_COLON:
            return {self.fixed["colon"]}
        if state == ParseState.EXPECT_FUNC_NAME_QUOTE:
            return {self.fixed["quote"]}
        if state == ParseState.EXPECT_FUNC_NAME:
            allowed = self.func_trie.next_tokens(self.func_name_tokens)
            if self.func_trie.is_complete(self.func_name_tokens):
                allowed.add(self.fixed["quote"])
            return allowed
        if state == ParseState.EXPECT_COMMA_AFTER_NAME:
            return {self.fixed["comma"]}
        if state == ParseState.EXPECT_PARAMS_KEY:
            if self.params_key_progress < len(self.params_key_tokens):
                return {self.params_key_tokens[self.params_key_progress]}
            return set()
        if state == ParseState.EXPECT_PARAMS_COLON:
            return {self.fixed["colon"]}
        if state == ParseState.EXPECT_PARAMS_OPEN:
            return {self.fixed["open_brace"]}
        if state == ParseState.EXPECT_PARAM_KEY:
            if self.param_key_progress < len(self.current_param_key_tokens):
                return {self.current_param_key_tokens[self.param_key_progress]}
            return set()
        if state == ParseState.EXPECT_PARAM_COLON:
            return {self.fixed["colon"]}
        if state == ParseState.EXPECT_VALUE_START:
            self.transition_to_value_state()
            return self.allowed_tokens()
        if state == ParseState.EXPECT_VALUE_QUOTE:
            return {self.fixed["quote"]}
        if state == ParseState.VALUE_NUMBER:
            if len(self.value_tokens) >= 20:
                allowed = {self.fixed["close_brace"]}
                if self.param_idx + 1 < len(self.param_list):
                    allowed.add(self.fixed["comma"])
                return allowed
            allowed = set(self._number_tokens)
            allowed.add(self.fixed["close_brace"])
            if self.param_idx + 1 < len(self.param_list):
                allowed.add(self.fixed["comma"])
            return allowed
        if state == ParseState.VALUE_STRING:
            if self._in_escape:
                return self._escape_token_ids | {self.fixed["quote"]}

            if len(self.value_tokens) >= self.max_string_tokens:
                return {self.fixed["quote"]}
            if self._current_enum_trie is not None:
                allowed = self._current_enum_trie.next_tokens(
                    self.value_tokens
                )
                if self._current_enum_trie.is_complete(self.value_tokens):
                    allowed.add(self.fixed["quote"])
                return allowed
            if self.current_param_format == "regex":
                return self._allowed_regex_tokens()
            if self.current_param_format == "replacement":
                if len(self.value_tokens) >= self.max_replacement_tokens:
                    return {self.fixed["quote"]}
                return set(self.string_safe_tokens) | {self.fixed["quote"]}
            base_allowed = set(self.string_safe_tokens) | {self.fixed["quote"]}
            if (
                len(self.value_tokens) == 0
                and self.backslash_token_id is not None
            ):
                base_allowed.discard(self.backslash_token_id)

            partial = self._decode_partial_string()
            if partial:
                quote_tok = self.fixed["quote"]
                extending = [
                    c for c in self._all_candidates
                    if len(c) > len(partial) and c.startswith(partial)
                ]
                if extending:
                    filtered = {
                        tid for tid in self.string_safe_tokens
                        if any(
                            c.startswith(
                                partial + self._cleaned_token_str.get(tid, "")
                            )
                            for c in extending
                        )
                    }
                    if filtered:
                        return filtered
                    if partial in self._quoted_spans:
                        return {quote_tok}
                    return set()
                if partial in self._quoted_spans:
                    return {quote_tok}

            return base_allowed
        if state == ParseState.VALUE_BOOLEAN:
            allowed = self.bool_trie.next_tokens(self.value_tokens)
            if self._is_valid_boolean():
                allowed.add(self.fixed["close_brace"])
                if self.param_idx + 1 < len(self.param_list):
                    allowed.add(self.fixed["comma"])
            return allowed
        if state == ParseState.EXPECT_PARAM_COMMA_OR_CLOSE:
            if self.param_idx + 1 < len(self.param_list):
                return {self.fixed["comma"]}
            return {self.fixed["close_brace"]}
        if state == ParseState.EXPECT_OBJECT_END:
            return {self.fixed["close_brace"]}
        if state == ParseState.DONE:
            return set()
        return set()

    # ------------------------------------------------------------------
    # Regex balance helpers
    # ------------------------------------------------------------------
    def _parse_regex_balance(self, raw: str) -> tuple[bool, int]:
        """Check whether a regex fragment has an open bracket or parentheses.

        Args:
            raw: The current regex string fragment.

        Returns:
            A tuple ``(bracket_open, paren_depth)`` indicating whether
            a character class is open and the current parenthesis depth.
        """
        bracket_open = False
        paren_depth = 0
        escape = False
        for ch in raw:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '[' and not bracket_open:
                bracket_open = True
            elif ch == ']' and bracket_open:
                bracket_open = False
            elif ch == '(' and not bracket_open:
                paren_depth += 1
            elif ch == ')' and not bracket_open and paren_depth > 0:
                paren_depth -= 1
        return bracket_open, paren_depth

    def _allowed_regex_tokens(self) -> Set[int]:
        """Compute allowed tokens for a regex string value.

        Tokens that would unbalance brackets or parentheses are excluded.

        Returns:
            Set of allowed token IDs for the current regex value.
        """
        if len(self.value_tokens) >= self.max_regex_tokens:
            return self._force_close_regex_tokens()
        raw = self._decode_partial_string()
        bracket_open, paren_depth = self._parse_regex_balance(raw)
        base = set(self.string_safe_tokens)
        disallowed: Set[int] = set()
        for tid in base:
            tok = self._cleaned_token_str.get(tid, "")
            if not bracket_open and ']' in tok:
                disallowed.add(tid)
            if not bracket_open and paren_depth == 0 and ')' in tok:
                disallowed.add(tid)
        allowed = base - disallowed
        if not bracket_open and paren_depth == 0:
            allowed.add(self.fixed["quote"])
        return allowed

    def _force_close_regex_tokens(self) -> Set[int]:
        """When the regex token limit is reached, force a balanced closure.

        Returns:
            A singleton set containing the required closing token.
        """
        raw = self._decode_partial_string()
        bracket_open, paren_depth = self._parse_regex_balance(raw)
        if bracket_open and self.regex_bracket_close_id is not None:
            return {self.regex_bracket_close_id}
        if paren_depth > 0 and self.regex_paren_close_id is not None:
            return {self.regex_paren_close_id}
        return {self.fixed["quote"]}

    def advance(self, token_id: int) -> None:
        """Advance the automaton state by consuming one token.

        This is the core transition function. It moves the state machine
        through the JSON skeleton, accumulates value tokens, and triggers
        parameter finalization when terminators are encountered.

        Args:
            token_id: The token ID consumed from the model output.

        Raises:
            RuntimeError: If an unexpected state transition is attempted.
        """
        #  State transition engine - advance(token_id) ---------------------
        # "Dato il token scelto dal modello (o iniettato), fa avanzare
        #  lo stato al prossimo."
        state = self.state
        if state == ParseState.EXPECT_OPEN_BRACE:
            self.state = ParseState.EXPECT_NAME_KEY
            self.name_key_progress = 0
        elif state == ParseState.EXPECT_NAME_KEY:
            self.name_key_progress += 1
            if self.name_key_progress >= len(self.name_key_tokens):
                self.state = ParseState.EXPECT_NAME_COLON
        elif state == ParseState.EXPECT_NAME_COLON:
            self.state = ParseState.EXPECT_FUNC_NAME_QUOTE
        elif state == ParseState.EXPECT_FUNC_NAME_QUOTE:
            self.state = ParseState.EXPECT_FUNC_NAME
        elif state == ParseState.EXPECT_FUNC_NAME:
            if token_id == self.fixed["quote"]:
                name = self._decode_name()
                self._set_function_name(name)
                self.state = ParseState.EXPECT_COMMA_AFTER_NAME
            else:
                self.func_name_tokens.append(token_id)
        elif state == ParseState.EXPECT_COMMA_AFTER_NAME:
            self.state = ParseState.EXPECT_PARAMS_KEY
            self.params_key_progress = 0
        elif state == ParseState.EXPECT_PARAMS_KEY:
            self.params_key_progress += 1
            if self.params_key_progress >= len(self.params_key_tokens):
                self.state = ParseState.EXPECT_PARAMS_COLON
        elif state == ParseState.EXPECT_PARAMS_COLON:
            self.state = ParseState.EXPECT_PARAMS_OPEN
        elif state == ParseState.EXPECT_PARAMS_OPEN:
            if not self.param_list:
                self.state = ParseState.EXPECT_OBJECT_END
            else:
                self.state = ParseState.EXPECT_PARAM_KEY
                self.param_key_progress = 0
                if self.param_idx < len(self.param_list):
                    pname = self.param_list[self.param_idx]
                    assert self.current_func is not None
                    self.current_param_key_tokens = (
                        self.param_key_ids[self.current_func.name][pname]
                    )
        elif state == ParseState.EXPECT_PARAM_KEY:
            self.param_key_progress += 1
            if self.param_key_progress >= len(self.current_param_key_tokens):
                self.state = ParseState.EXPECT_PARAM_COLON
                self.current_param_name = self.param_list[self.param_idx]
                assert self.current_func is not None
                self.current_param_type = self.current_func.parameters[
                    self.current_param_name
                ].type
                self.current_param_format = self.param_formats.get(
                    (self.current_func.name, self.current_param_name), None
                )
                self.value_tokens = []
        elif state == ParseState.EXPECT_PARAM_COLON:
            self.state = ParseState.EXPECT_VALUE_START
        elif state == ParseState.EXPECT_VALUE_START:
            raise RuntimeError(
                "EXPECT_VALUE_START should not be reached in advance()"
            )
        elif state == ParseState.EXPECT_VALUE_QUOTE:
            self.state = ParseState.VALUE_STRING
        elif state == ParseState.VALUE_NUMBER:
            if token_id in (self.fixed["comma"], self.fixed["close_brace"]):
                self._finish_param_value()
                if token_id == self.fixed["comma"]:
                    self.param_idx += 1
                    if self.param_idx >= len(self.param_list):
                        self.state = ParseState.EXPECT_OBJECT_END
                        return
                    self.state = ParseState.EXPECT_PARAM_KEY
                    self.param_key_progress = 0
                    pname = self.param_list[self.param_idx]
                    assert self.current_func is not None
                    self.current_param_key_tokens = (
                        self.param_key_ids[self.current_func.name][pname]
                    )
                else:
                    self.state = ParseState.EXPECT_OBJECT_END
            else:
                self.value_tokens.append(token_id)
        elif state == ParseState.VALUE_STRING:
            if self._in_escape:
                self.value_tokens.append(token_id)
                self._in_escape = False
                return
            if token_id == self.fixed["quote"]:
                self._finish_param_value()
                self.state = ParseState.EXPECT_PARAM_COMMA_OR_CLOSE
                self._current_enum_trie = None
            else:
                self.value_tokens.append(token_id)
                cleaned_raw = self._cleaned_raw_value()
                bs_count = 0
                for ch in reversed(cleaned_raw):
                    if ch == '\\':
                        bs_count += 1
                    else:
                        break
                self._in_escape = (bs_count % 2 == 1)
        elif state == ParseState.VALUE_BOOLEAN:
            if token_id in (self.fixed["comma"], self.fixed["close_brace"]):
                self._finish_param_value()
                if token_id == self.fixed["comma"]:
                    self.param_idx += 1
                    if self.param_idx >= len(self.param_list):
                        self.state = ParseState.EXPECT_OBJECT_END
                        return
                    self.state = ParseState.EXPECT_PARAM_KEY
                    self.param_key_progress = 0
                    pname = self.param_list[self.param_idx]
                    assert self.current_func is not None
                    self.current_param_key_tokens = (
                        self.param_key_ids[self.current_func.name][pname]
                    )
                else:
                    self.state = ParseState.EXPECT_OBJECT_END
            else:
                self.value_tokens.append(token_id)
        elif state == ParseState.EXPECT_PARAM_COMMA_OR_CLOSE:
            if token_id == self.fixed["comma"]:
                self.param_idx += 1
                if self.param_idx >= len(self.param_list):
                    self.state = ParseState.EXPECT_OBJECT_END
                    return
                self.state = ParseState.EXPECT_PARAM_KEY
                self.param_key_progress = 0
                pname = self.param_list[self.param_idx]
                assert self.current_func is not None
                self.current_param_key_tokens = (
                    self.param_key_ids[self.current_func.name][pname]
                )
            elif token_id == self.fixed["close_brace"]:
                self.state = ParseState.EXPECT_OBJECT_END
        elif state == ParseState.EXPECT_OBJECT_END:
            self.state = ParseState.DONE
        elif state == ParseState.DONE:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _decode_name(self) -> str:
        """Decode the accumulated function name tokens into a string.

        Returns:
            The cleaned function name.
        """
        raw = "".join(self.vocab.get(t, "") for t in self.func_name_tokens)
        return self._clean_string_value(raw)

    def _set_function_name(self, name: str) -> None:
        """Set the active function and prepare its parameter list.

        Args:
            name: The decoded function name.

        Raises:
            ValueError: If there are no function definitions available.
        """
        if name not in self.func_defs:
            logger.error("Unknown function name decoded: '%s'", name)
            if not self.func_defs:
                raise ValueError("No function definitions available")
            name = next(iter(self.func_defs))
            logger.warning("Falling back to function '%s'", name)
        self.current_func = self.func_defs[name]
        self.param_list = list(self.current_func.parameters.keys())
        self.param_idx = 0
        self.param_key_progress = 0

    def _is_valid_boolean(self) -> bool:
        """Check whether the current value tokens form a complete boolean.

        Returns:
            True if the accumulated tokens decode to ``"true"`` or
            ``"false"``.
        """
        decoded = "".join(self.vocab.get(tid, "") for tid in self.value_tokens)
        return decoded.lower() in ("true", "false")

    @staticmethod
    def _clean_string_value(raw: str) -> str:
        """Normalize a raw string value by removing special markers.

        Args:
            raw: Raw string that may contain ``Ġ`` or ``Ċ`` markers.

        Returns:
            The cleaned and stripped string.
        """
        return raw.replace("Ġ", " ").replace("Ċ", " ").strip()

    def _finish_param_value(self) -> None:
        """Parse the accumulated value tokens and store the result.

        The interpretation depends on the declared parameter type.
        Default values are used when the token buffer is empty.
        """
        ptype = self.current_param_type
        if not self.value_tokens:
            if ptype == "integer":
                self.param_values.append(0)
            elif ptype == "number":
                self.param_values.append(0.0)
            elif ptype == "boolean":
                self.param_values.append(False)
            elif ptype == "string":
                self.param_values.append("")
            else:
                self.param_values.append(None)
            return
        if ptype == "integer":
            raw = "".join(
                self.vocab.get(tid, "") for tid in self.value_tokens
            )
            try:
                self.param_values.append(int(float(raw)))
            except ValueError:
                self.param_values.append(0)
        elif ptype == "number":
            raw = "".join(
                self.vocab.get(tid, "") for tid in self.value_tokens
            )
            try:
                self.param_values.append(float(raw))
            except ValueError:
                self.param_values.append(0.0)
        elif ptype == "string":
            raw = "".join(
                self.vocab.get(t, "") for t in self.value_tokens
            )
            s = self._clean_string_value(raw)
            s = s.replace("\\\\", "\\")
            s = s.replace('\\"', '"')
            s = s.replace('\\/', '/')
            while s.endswith("}}"):
                s = s[:-2]
            s = s.rstrip('_').strip()
            s = ''.join(
                ch for ch in s if ord(ch) >= 32 or ch == '\n'
            )
            self.param_values.append(s)
        elif ptype == "boolean":
            raw = "".join(
                self.vocab.get(tid, "") for tid in self.value_tokens
            )
            self.param_values.append(raw.lower() == "true")
        else:
            raw = "".join(
                self.vocab.get(tid, "") for tid in self.value_tokens
            )
            self.param_values.append(raw)

    def force_close_current_value(self) -> None:
        """Force-finalize the current parameter value (recovery logic).

        This is called when no allowed tokens are available; it appends a
        default value and advances the state machine to the next parameter
        or the end of the object.
        """
        #  Chiusura Forzata - force_close_current_value() ------------------
        # "Il meccanismo di recovery del Bonus 2: chiude forzatamente
        #  un valore bloccato, aggiungendo i caratteri sintattici
        #  mancanti."
        state = self.state
        if state == ParseState.VALUE_STRING:
            self._finish_param_value()
            self.state = ParseState.EXPECT_PARAM_COMMA_OR_CLOSE
        elif state in (ParseState.VALUE_NUMBER, ParseState.VALUE_BOOLEAN):
            self._finish_param_value()
            if self.param_idx + 1 < len(self.param_list):
                self.param_idx += 1
                self.state = ParseState.EXPECT_PARAM_KEY
                self.param_key_progress = 0
                if self.current_func is not None:
                    pname = self.param_list[self.param_idx]
                    self.current_param_key_tokens = self.param_key_ids[
                        self.current_func.name][pname]
            else:
                self.state = ParseState.EXPECT_OBJECT_END
        else:
            logger.warning(
                "force_close_current_value in unexpected state: %s",
                state
            )

    def is_complete(self) -> bool:
        """Check whether generation has reached the DONE state.

        Returns:
            True if the automaton has finished.
        """
        return self.state == ParseState.DONE

    def get_result(self) -> Dict[str, Any]:
        """Package the accumulated function name and parameter values.

        Returns:
            A dictionary with keys ``"name"`` and ``"parameters"``.
            If generation is incomplete, an ``"_error"`` key is added.
        """
        params = {}
        if self.current_func:
            for i, pname in enumerate(self.param_list):
                if i < len(self.param_values):
                    params[pname] = self.param_values[i]
                else:
                    logger.warning(
                        "Missing value for parameter '%s'",
                        pname
                    )
                    params[pname] = None
        result = {
            "name": self.current_func.name if self.current_func else "",
            "parameters": params,
        }
        if not self.is_complete():
            result["_error"] = (
                f"Incomplete generation in state {self.state.name}"
            )
        return result
