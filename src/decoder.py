"""Constrained decoding with automatic JSON structure injection."""
from __future__ import annotations

import logging
from typing import Any, Set
import numpy as np
import numpy.typing as npt

from .model_protocol import ModelProtocol
from .models import FunctionDef
from .schema import JSONStateManager, ParseState
from .token_utils import TokenTrie
from .tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)


class ConstrainedDecoder:
    """Decoder that guides token-by-token generation by injecting JSON syntax.

    This decoder enforces a strict JSON structure during autoregressive text
    generation. It uses a state machine to track the expected JSON tokens and
    restricts the model's output to only those tokens that are syntactically
    valid at each step. It supports function calling with strongly typed
    parameters, including enumerations and booleans.

    Attributes:
        model: The underlying language model exposing
            ``get_logits_from_input_ids``.
        vocab: Mapping from token id to string token.
        funcs: Dictionary of available functions, keyed by name.
        tokenizer: Custom tokenizer used for encoding/decoding.
        max_tokens: Maximum number of tokens to generate in one call.
        temperature: Sampling temperature (0.0 for greedy decoding).
        fixed: Pre-computed single-token ids for structural JSON characters.
        func_trie: Trie containing tokenized function names.
        enum_tries: Mapping ``(func_name, param_name) -> TokenTrie`` for
            allowed enum values.
        bool_trie: Trie containing the tokens for ``true`` and ``false``.
        param_key_ids: Mapping ``func_name -> param_name -> list[int]``
            for parameter key tokens.
        _string_safe_tokens: Set of token ids considered safe for use
            inside string values.
        _number_tokens: Set of token ids that represent numeric literals.
        _prompt_cache: Cache of previously generated results keyed by prompt.
        _prompt_prefix_ids: Encoded system prompt prefix.
        _json_suffix_ids: Encoded suffix that precedes the JSON response.
    """

    def __init__(
        self,
        model: ModelProtocol,
        vocab: dict[int, str],
        functions: list[FunctionDef],
        tokenizer: CustomTokenizer,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> None:
        """Initialize the ConstrainedDecoder.

        Args:
            model: The underlying model implementing
                ``get_logits_from_input_ids``.
            vocab: Vocabulary mapping token IDs to string representations.
            functions: List of ``FunctionDef`` objects describing the
                available callable functions.
            tokenizer: A custom tokenizer with ``encode`` and ``decode``
                methods.
            max_tokens: Maximum number of generation steps.
            temperature: Sampling temperature; 0.0 means greedy selection.

        Raises:
            RuntimeError: If any required structural token cannot be encoded.
        """
        self.model = model
        self.vocab = vocab
        self.funcs = {f.name: f for f in functions}
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature

        # ---------------------------------------------------------------------
        # Helper to encode text with the custom tokenizer
        # ---------------------------------------------------------------------
        def enc(text: str) -> list[int]:
            return self.tokenizer.encode(text)

        # ---- Fixed structural token IDs (must exist) ------------------------
        self.fixed: dict[str, int] = {}
        for name, token_str in [
            ("open_brace", "{"),
            ("close_brace", "}"),
            ("quote", '"'),
            ("colon", ":"),
            ("comma", ","),
        ]:
            ids = enc(token_str)
            if not ids:
                raise RuntimeError(
                    f"Required token '{token_str}' not found."
                )
            self.fixed[name] = ids[0]
        self.name_key_ids: list[int] = enc('"name"')
        self.params_key_ids: list[int] = enc('"parameters"')
        self.func_trie = TokenTrie()
        self.enum_tries: dict[tuple[str, str], TokenTrie] = {}
        for fname, fdef in self.funcs.items():
            for pname, pdef in fdef.parameters.items():
                if pdef.allowed_values:
                    trie = TokenTrie()
                    for value in pdef.allowed_values:
                        trie.insert(self.tokenizer.encode(value))
                    self.enum_tries[(fname, pname)] = trie
        for name in self.funcs:
            self.func_trie.insert(enc(name))
        self.bool_trie = TokenTrie()
        self.bool_trie.insert(enc("true"))
        self.bool_trie.insert(enc("false"))
        # ---------------------------------------------------------------------
        # Pre-encode parameter keys per function
        # ---------------------------------------------------------------------
        self.param_key_ids: dict[str, dict[str, list[int]]] = {}
        for fname, fdef in self.funcs.items():
            self.param_key_ids[fname] = {}
            for pname in fdef.parameters:
                self.param_key_ids[fname][pname] = enc(f'"{pname}"')
        # ---------------------------------------------------------------------
        # Build safe-token sets for value generation
        # ---------------------------------------------------------------------
        # String-safe tokens: exclude tokens containing quotes, braces,
        # backticks, angle brackets, pure dots, or non-ASCII control chars
        # ---------------------------------------------------------------------
        self._string_safe_tokens: Set[int] = set()
        for tid, tok in self.vocab.items():
            if not tok:
                continue
            if '"' in tok:
                continue
            if '{' in tok and '}' in tok:
                continue
            if '`' in tok:
                continue
            if all(c == '.' for c in tok):
                continue
            if '<' in tok or '>' in tok:
                continue
            cleaned = tok.replace("Ġ", " ").replace("Ċ", "\n")
            if any(ord(ch) < 32 or ord(ch) > 126 for ch in cleaned):
                continue
            self._string_safe_tokens.add(tid)
        # Number tokens: tokens that consist only of digits and numeric chars -
        self._number_tokens: Set[int] = {
            tid for tid, tok in self.vocab.items()
            if tok and all(c in "0123456789.-+eE" for c in tok)
        }
        # Cache for prompt results and build the internal prompt template -----
        self._prompt_cache: dict[str, dict[str, Any]] = {}
        self._build_prompt_template()

    def _build_prompt_template(self) -> None:
        """Build the internal system prompt and pre-encode its parts.

        The prompt template instructs the model to output a JSON object
        with exactly two keys: ``"name"`` and ``"parameters"``, and lists
        all available functions with their parameter types.
        """
        system = (
            "You are a precise function calling assistant. "
            "Output a JSON object with exactly two keys: "
            "\"name\" and \"parameters\".\n"
            "Never output type names as values.\n\n"
            "Functions:\n"
        )
        func_lines = []
        for f in self.funcs.values():
            params_desc = ", ".join(
                f"{n}=<{i.type.upper()}>" for n, i in f.parameters.items()
            )
            func_lines.append(
                f"  {f.name}: {f.description} | Args: {params_desc}"
            )
        system += "\n".join(func_lines)
        # ---------------------------------------------------------------------
        # Prefix: system message + "User: "; suffix: "\nJSON: "
        # ---------------------------------------------------------------------
        self._prompt_prefix_ids = self.tokenizer.encode(system + "\n\nUser: ")
        self._json_suffix_ids = self.tokenizer.encode("\nJSON: ")

    @staticmethod
    def _sample_token(
        logits: npt.NDArray[np.floating], temperature: float
    ) -> int:
        """Sample a token ID from a logits array using the given temperature.

        If the temperature is <= 0 or all logits are -inf, greedy decoding
        is used (argmax). Otherwise, softmax probabilities are computed and
        a token is sampled randomly according to those probabilities.

        Args:
            logits: 1D array of unnormalized log-probabilities.
            temperature: Sampling temperature.

        Returns:
            The selected token ID.
        """
        if temperature <= 0.0 or np.max(logits) == -np.inf:
            return int(np.argmax(logits))

        probs = np.exp(logits / temperature)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def generate(self, prompt: str) -> dict[str, Any]:
        """Generate a constrained JSON function call in response to a prompt.

        The prompt is automatically prefixed with a system instruction and a
        ``"JSON: "`` suffix before generation. The result is cached to avoid
        re-decoding identical prompts.

        Args:
            prompt: The user prompt string.

        Returns:
            A dictionary representing the generated JSON, typically with keys
            ``"name"`` and ``"parameters"``. If an error occurs during
            generation, an ``"_error"`` key is removed before returning.
        """
        #  Genearate: La fase‑ponte tra Decoder e Json: -----------------------
        # "decoder.py guida il loop, schema.py decide cosa è permesso in ogni
        #  istante di quel loop.
        if prompt in self._prompt_cache:
            logger.debug(
                "Cache hit for prompt: %s",
                prompt[:80]
            )
            return self._prompt_cache[prompt].copy()
        logger.debug("=" * 60)
        logger.debug(
            "Processing prompt: %s",
            prompt[:120]
        )
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = (self._prompt_prefix_ids +
                     prompt_ids + self._json_suffix_ids)
        logger.debug(
            "Tokenised: %d total",
            len(input_ids)
        )
        logger.debug(
            "Starting constrained generation (temperature=%.2f)...",
            self.temperature
        )
        # Instantiate the state machine with all lookup structures ------------
        state_mgr = JSONStateManager(
            func_trie=self.func_trie,
            func_defs=self.funcs,
            fixed_tokens=self.fixed,
            param_key_ids=self.param_key_ids,
            bool_trie=self.bool_trie,
            string_safe_tokens=self._string_safe_tokens,
            name_key_ids=self.name_key_ids,
            enum_tries=self.enum_tries,
            params_key_ids=self.params_key_ids,
        )
        state_mgr.set_vocab(self.vocab)
        state_mgr.set_prompt(prompt)
        step = 0
        recovery_attempts = 0
        while step < self.max_tokens and state_mgr.state != ParseState.DONE:
            #   Inject any structural tokens required by the current state ----
            #   1. _inject_structural – se lo stato è EXPECT_*
            self._inject_structural(state_mgr, input_ids)
            if state_mgr.is_complete():
                break
            #   Get model logits for the current prefix -----------------------
            #   2. self.model.get_logits_from_input_ids(input_ids) – se lo
            #      stato è VALUE_*, qui si chiede al modello i logit
            logits_raw = self.model.get_logits_from_input_ids(input_ids)
            logits_np = np.array(logits_raw)
            #   3. allowed = state_mgr.allowed_tokens() – chiede a schema.py --
            #      cosa è ammesso ORA.
            allowed = state_mgr.allowed_tokens()
            if not allowed:
                recoverable = {
                    ParseState.VALUE_STRING,
                    ParseState.VALUE_NUMBER,
                    ParseState.VALUE_BOOLEAN
                }
                if state_mgr.state in recoverable and recovery_attempts < 3:
                    recovery_attempts += 1
                    logger.warning(
                        "Recovery attempt %d: force closing value",
                        recovery_attempts
                    )
                    state_mgr.force_close_current_value()
                    continue
                else:
                    logger.error(
                        "No allowed tokens at step %d, state %s",
                        step, state_mgr.state
                    )
                    break
            else:
                recovery_attempts = 0
            #  Mask logits to keep only allowed tokens ------------------------
            #  4. mask a -inf – tutto ciò che non è in allowed sparisce dalla
            #     scelta del modello. (softmax. e^(-inf) = 0)
            mask = np.full_like(logits_np, -np.inf)
            mask[list(allowed)] = 0.0
            masked = logits_np + mask
            next_id = self._sample_token(masked, self.temperature)
            # Debug logging for detailed step tracking ------------------------
            if logger.isEnabledFor(logging.DEBUG):
                token_text = self.vocab.get(next_id, f"<ID:{next_id}>")
                value_preview = ""
                if state_mgr.value_tokens:
                    raw = "".join(
                        self.vocab.get(tid, "?")
                        for tid in state_mgr.value_tokens[-10:]
                    )
                    value_preview = raw.replace("\n", "\\n")[:40]

                func_name = (
                    state_mgr.current_func.name
                    if state_mgr.current_func
                    else "(building)"
                )

                param_name = state_mgr.current_param_name or "-"
                logger.debug("Step %3d | token: %-12s | state: %-25s | "
                             "func: %-15s | param: %-10s | value: %s",
                             step,
                             repr(token_text)[:12],
                             state_mgr.state.name,
                             func_name[:15],
                             param_name[:10],
                             value_preview)
            # Advance the state machine with the chosen token ----------------
            #   5. state_mgr.advance(next_id) – lo stato avanza, si ricomincia
            #      dal punto 1."
            state_mgr.advance(next_id)
            input_ids.append(next_id)
            step += 1
        logger.debug(
            "Generation completed in %d steps",
            step
        )
        result = state_mgr.get_result()
        result.pop("_error", None)
        self._prompt_cache[prompt] = result
        return result

    def _inject_structural(
            self, state_mgr: JSONStateManager, input_ids: list[int]
    ) -> None:
        """Inject structural tokens required by the current parse state.

        This method runs in a loop, appending deterministic tokens (braces,
        quotes, colons, commas, keys) until the state machine reaches a
        state where the model must choose the next token (function name,
        parameter value, etc.).

        Args:
            state_mgr: The current state manager.
            input_ids: Mutable list of token IDs being built.
        """
        #  _inject_structural -------------------------------------------------
        # "Se lo stato è EXPECT_*, inietta il carattere strutturale
        #  direttamente, il modello non lo vede nemmeno come scelta." ---------
        while True:
            state = state_mgr.state
            # Structural injection based on current parse state ---------------
            if state == ParseState.EXPECT_OPEN_BRACE:
                self._append_and_advance(
                    input_ids, state_mgr, self.fixed["open_brace"]
                )
            elif state == ParseState.EXPECT_NAME_KEY:
                self._append_and_advance_multi(
                    input_ids, state_mgr, self.name_key_ids
                )
            elif state == ParseState.EXPECT_NAME_COLON:
                self._append_and_advance(
                    input_ids, state_mgr, self.fixed["colon"]
                )
            elif state == ParseState.EXPECT_FUNC_NAME_QUOTE:
                self._append_and_advance(
                    input_ids, state_mgr, self.fixed["quote"]
                )
            elif state == ParseState.EXPECT_FUNC_NAME:
                break
            elif state == ParseState.EXPECT_COMMA_AFTER_NAME:
                self._append_and_advance(
                    input_ids, state_mgr, self.fixed["comma"]
                )
            elif state == ParseState.EXPECT_PARAMS_KEY:
                self._append_and_advance_multi(
                    input_ids, state_mgr, self.params_key_ids
                )
            elif state == ParseState.EXPECT_PARAMS_COLON:
                self._append_and_advance(
                    input_ids, state_mgr, self.fixed["colon"]
                )
            elif state == ParseState.EXPECT_PARAMS_OPEN:
                self._append_and_advance(
                    input_ids, state_mgr, self.fixed["open_brace"]
                )
                state_mgr.prepare_first_param()
            elif state == ParseState.EXPECT_PARAM_KEY:
                current_key_ids = state_mgr.get_current_param_key_tokens()
                self._append_and_advance_multi(
                    input_ids, state_mgr, current_key_ids
                )
            elif state == ParseState.EXPECT_PARAM_COLON:
                self._append_and_advance(
                    input_ids, state_mgr, self.fixed["colon"]
                )
            elif state == ParseState.EXPECT_VALUE_START:
                state_mgr.transition_to_value_state()
                if state_mgr.state == ParseState.EXPECT_VALUE_QUOTE:
                    self._append_and_advance(
                        input_ids, state_mgr, self.fixed["quote"]
                    )
                elif state_mgr.state in (
                    ParseState.VALUE_NUMBER,
                    ParseState.VALUE_BOOLEAN
                ):
                    break
                else:
                    if state_mgr.state == ParseState.VALUE_STRING:
                        break
            #  tutti i casi EXPECT_* con iniezione diretta --------------------
            elif state in (
                ParseState.VALUE_NUMBER,
                ParseState.VALUE_STRING,
                ParseState.VALUE_BOOLEAN
            ):
                break  # Model is generating the value content ----------------
            elif state == ParseState.EXPECT_PARAM_COMMA_OR_CLOSE:
                if state_mgr.param_idx + 1 < len(state_mgr.param_list):
                    self._append_and_advance(
                        input_ids, state_mgr, self.fixed["comma"]
                    )
                else:
                    self._append_and_advance(
                        input_ids, state_mgr, self.fixed["close_brace"]
                    )
            elif state == ParseState.EXPECT_OBJECT_END:
                self._append_and_advance(
                    input_ids, state_mgr, self.fixed["close_brace"]
                )
            elif state == ParseState.DONE:
                break
            else:
                break

    def _append_and_advance(
            self,
            input_ids: list[int],
            sm: JSONStateManager,
            token_id: int
    ) -> None:
        """Append a single structural token and advance the state machine.

        Args:
            input_ids: The token list to append to.
            sm: The state manager to advance.
            token_id: The token ID to append.
        """
        input_ids.append(token_id)
        sm.advance(token_id)

    def _append_and_advance_multi(
            self,
            input_ids: list[int],
            sm: JSONStateManager,
            token_ids: list[int]
    ) -> None:
        """Append a sequence of structural tokens and advance the state.

        Args:
            input_ids: The token list to append to.
            sm: The state manager to advance.
            token_ids: The list of token IDs to append sequentially.
        """
        for tid in token_ids:
            self._append_and_advance(input_ids, sm, tid)
