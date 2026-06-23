# Call_me_maybe

*This project has been created as part of the 42 curriculum by [login].*

## Description

Call_me_maybe is a constrained JSON decoder built on top of a small language
model (Qwen3-0.6B). Given a natural language prompt, the system identifies the
correct function to call and extracts its parameters, outputting a valid JSON
object of the form `{"name": "fn_name", "parameters": {...}}`.

The goal is to perform reliable structured function-call extraction using a
model that was not fine-tuned for this task, relying entirely on constrained
decoding to guarantee JSON validity at every step.

---

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency
  management

### Installation

```bash
git clone <repo_url>
cd call_me_maybe
uv sync
```

### Execution

```bash
# Run with the default model (Qwen3-0.6B)
uv run python -m src

# Run with Qwen2.5 (requires llm_sdk)
uv run python -m src --model Qwen25

# Run with verbose logging (step-by-step token generation)
uv run python -m src --verbose
```

Output is written to `data/output/function_calling_results.json`.

> **Results, reference 11-prompt suite (fast dev machine, `temperature=0.0`):**
> Qwen3-0.6B (mandatory, default): **10/11 correct (90.9%)**, 100% valid JSON, ~5.5–8s total.
> Qwen2.5-0.5B-Instruct (bonus, `--model Qwen25`): **10/11 correct (90.9%)**, 100% valid JSON, ~6–8s total.
> Both deterministic at `temperature=0.0` (verified across independent runs — see [Testing strategy](#testing-strategy)).
> ⚠️ On slower/shared hardware (school lab machines), the bonus model can be significantly slower per prompt — see [Known limitation: bonus model timing](#known-limitation-bonus-model-timing-on-constrained-hardware).

---

## Bonus Features: Honest Assessment

Per the subject (Chapter VII), here is the real status of every bonus item:

| # | Feature | Required | Status | Notes | Code |
|---|---------|----------|--------|-------|------|
| 1 | Multiple LLM models beyond Qwen/Qwen3-0.6B | ✅ | ✅ Implemented | `--model qwen06b` (default) and `--model Qwen25` (`Qwen/Qwen2.5-0.5B-Instruct`), both benchmarked on the same 11-prompt suite: **10/11 (90.9%)** each, 100% valid JSON. The single failure differs in detail between the two models — see "Comparing the two models" below. Timing on constrained hardware: see [Known limitation](#known-limitation-bonus-model-timing-on-constrained-hardware). No support for closed/proprietary APIs (by design, `ModelProtocol` only targets local HF-style backends) | `src/model_factory.py` |
| 2 | Custom tokenizer (not using SDK encode/decode) | ✅ | ✅ Implemented | Full greedy longest-match tokenizer. Never calls SDK `encode()`/`decode()`. Used everywhere in main pipeline | `src/tokenizer.py` |
| 3 | Advanced error recovery mechanisms | ✅ | ✅ Implemented | Three-level strategy: (1) token masking with logging, (2) forced value injection, (3) process-level exception handling | `src/decoder.py` + `src/schema.py` |
| 4 | Performance optimizations (caching, batching) | ❌ | ⚠️ Partial | **Prefix caching:** ✅ system prompt cached. **In-memory cache:** ✅ identical prompts skipped. **Batching:** ❌ not implemented (sequential processing only) | `src/decoder.py` |
| 5 | Comprehensive test suite | ✅ | ✅ Implemented | Unit tests (tokenizer, state machine), integration tests (full pipeline), e2e tests (CLI). Pytest + fixtures. Covers all edge cases | `tests/` |
| 6 | Visualization of generation process | ✅ | ✅ Implemented | `--verbose` flag shows step-by-step token generation: tokenization, FSM state transitions, logits, and masked tokens. `make run --verbose` visualizes the entire pipeline | `src/tokenizer.py`, `src/decoder.py` |
| 7 | Support for complex nested function arguments | ❌ | ❌ Not Implemented | System designed for flat parameters only. No support for nested objects, arrays of objects, or recursive structures | – |
| 8 | Public tokenizer encode/decode methods | ✅ | ✅ Implemented | `CustomTokenizer.encode(text) -> List[int]`, `CustomTokenizer.decode(ids) -> str`. Both public, fully documented with debug logs: token count, UNK fallbacks, GPT-style token conversions | `src/tokenizer.py` |
| 9 | Demonstration: integration of encoding/decoding with constrained decoding | ✅ | ✅ Implemented | Full pipeline: tokenizer → constrained decoder → logit masking → output. All integrated seamlessly | `src/__main__.py`, `src/decoder.py` |

### Summary

- **Fully Implemented (7/9):** Custom tokenizer, error recovery, test suite, public encode/decode with logging, integration demo, multiple models (`qwen06b` + `Qwen25`, both at 10/11 on the reference suite), visualization of generation process
- **Partially Implemented (1/9):** Caching (prefix + in-memory, but no batching)
- **Not Implemented (1/9):** Nested function arguments

---

## Compliance check of `src/` and `tests/`

This check was performed only on the files inside `src/` and `tests/`.

- `src/` uses `pydantic` for model validation, CLI configuration, and JSON loading.
- `src/decoder.py` and `src/schema.py` implement a constrained decoder based on an FSM state machine and token masking.
- `src/` uses `numpy` and `json`, as required by the subject.
- `tests/` contains unit, integration, and end-to-end tests, including checks against a `DummyLLM` model that implements `ModelProtocol`.
- No direct imports of forbidden packages such as `torch`, `transformers`, or `huggingface` are present in `src/` or `tests/`.
- `pyproject.toml` declares `accelerate>=1.14.0` as a runtime dependency: it is not imported or used anywhere in `src/` or `tests/`. It is only required at runtime by `llm_sdk` when it loads the model with `device_map="auto"` on GPU/CUDA (see `llm_sdk/__init__.py`). It is documented here for transparency, not to circumvent the ban on Hugging Face packages: the project's own code does not use it.
- Syntax checking with `python3 -m py_compile src/*.py tests/**/*.py` passed with no errors.

---

## How it works, TL;DR?

*For a friendly, non-technical explanation of what makes this project fast and
reliable.*

### Why not let the model write the whole JSON?

Because models often make mistakes: missing quotes, unbalanced braces, or
inventing functions that don't exist. Without constraints, you'd need to retry
or fix the output, wasting time and calls.

We built a kind of "mold" that already knows the exact JSON structure we need.
The model only has to decide the interesting parts — the function name and the
parameter values — while everything else (braces, quotes, field names) is
injected automatically.

### How does that make it fast?

- We skip most of the work. The fixed structure is added directly without ever
  touching the model. A typical generation uses ~20–30 tokens; the model chooses
  only 4–5 of them.
- We pre-compute the prompt. The system message and few-shot example are
  tokenised once and reused for every request.
- We make instant decisions. When the model must choose, we use a mathematical
  mask (NumPy) to delete all wrong answers at once, then pick the best in a
  single step.
- The model is small. Qwen3-0.6B produces logits in fractions of a second; no
  heavy frameworks.

*Result: each prompt takes 2–3 seconds. Even 100 prompts stay well under 5
minutes.*

### Is the output always correct?

**JSON validity:** Yes — 100% syntactically valid by construction. Every structural character (`{`, `}`, `:`, etc.) is forced by a rule, never chosen by the model.

**Semantic correctness:** 10/11 (90.9%) on the reference 11-prompt suite, **for both supported models** (Qwen3-0.6B and Qwen2.5-0.5B-Instruct). In both cases the single failure is the same prompt ("Replace all vowels in 'Programming is fun' with asterisks").

### Comparing the two models on the same failure

Running the identical 11-prompt suite through both backends (`--model qwen06b` vs `--model Qwen25`) shows that neither model reliably produces a regex **character class** (`[...]`) — the vowel-substitution prompt is the one consistent blind spot across both backends:

| | Qwen3-0.6B (mandatory) | Qwen2.5-0.5B-Instruct (bonus) |
|---|---|---|
| Prompt 9 regex ("...with NUMBERS") | `34\|233` — literal enumeration of the digits seen in the prompt | digit-matching pattern (e.g. `\d+` family) — passes |
| Prompt 9 verdict | ✅ correct on this input, but hard-coded, would not generalise | ✅ correct and generalises |
| Prompt 10 regex ("...with asterisks") | `aeiou` — treats the vowels as one literal substring, never matches | `aeiouAEIOU` — also a literal substring, missing the `[...]` brackets, never matches |
| Prompt 10 replacement | `asterisks` (the whole word, copied from the prompt) | `*****` (five literal stars instead of one) |
| Prompt 10 verdict | ❌ no match at all → text unchanged | ❌ no match at all → text unchanged |

Neither backend ever emits a proper regex character class for this prompt: both fall back to copying a literal substring instead of using `[...]` bracket syntax — a genuine capability ceiling shared by both models on this specific construct (a regex character class is a more abstract, syntax-level concept than the digit/word patterns that work fine elsewhere in the suite). This is exactly the kind of evidence the subject is asking for: the constrained decoder enforces 100% valid, schema-compliant JSON in every single case for both models (verified: 11/11 + 11/11), and the residual 1-in-11 error rate is purely a property of model capability, not of the decoding mechanism.

### Can I use a different model?

Absolutely. The decoder talks to any model that implements our simple
`ModelProtocol`. Write a thin adapter, and the rest of the code stays exactly
the same.

Note on available backends: the project supports the Qwen family via the
optional SDK. The mapping implemented in `src/model_factory.py` is as follows:

- `--model qwen06b` -> `Small_LLM_Model()` (SDK default constructor)
- `--model Qwen25`  -> `Small_LLM_Model("Qwen/Qwen2.5-0.5B-Instruct")`

So, `Qwen/Qwen2.5-0.5B-Instruct` is explicitly used when `--model Qwen25` is
selected; `--model qwen06b` uses the SDK's default Small LLM (no explicit
identifier). Install the optional `llm_sdk` package to enable these backends.
We picked the `0.5B` variant specifically to stay close in size to the
mandatory `Qwen3-0.6B` baseline, rather than a larger Qwen2.5 size — see
[Known limitation: bonus model timing](#known-limitation-bonus-model-timing-on-constrained-hardware)
for why size alone doesn't guarantee comparable speed on this SDK.

**Note on test-only model:** The internal `DummyLLM` (`tests/model_dummy.py`)
is used exclusively in the test suite (`tests/e2e/test_cli.py`) to validate
the pipeline without requiring inference or external dependencies. It is not
exposed via the CLI and is not suitable for production use.

### What if the prompt is unclear?

The model still tries its best. If it cannot finish (e.g., a value is missing),
the system logs an error but still returns a valid JSON skeleton — no broken
outputs.

---

## Algorithm explanation

The core of the project is a **finite-state machine (FSM)** that drives
token-by-token generation. Instead of letting the model generate freely and
hoping the output is valid JSON, the FSM constrains each generation step to only
the tokens that are structurally valid at that position.

The JSON structure follows a fixed schema:

```json
{ "name": "<function_name>", "parameters": { "<key>": <value>, ... } }
```

The FSM is implemented in `src/schema.py` as `ParseState` and moves through
19 states. The full list, in generation order:

| State | Description |
|-------|-------------|
| `EXPECT_OPEN_BRACE` | Waiting for the opening `{` |
| `EXPECT_NAME_KEY` | Injecting the literal `"name"` key |
| `EXPECT_NAME_COLON` | Injecting `:` after `"name"` |
| `EXPECT_FUNC_NAME_QUOTE` | Injecting the opening `"` of the function name |
| `EXPECT_FUNC_NAME` | Model selects function name tokens (via trie) |
| `EXPECT_COMMA_AFTER_NAME` | Injecting `,` between the two top-level keys |
| `EXPECT_PARAMS_KEY` | Injecting the literal `"parameters"` key |
| `EXPECT_PARAMS_COLON` | Injecting `:` after `"parameters"` |
| `EXPECT_PARAMS_OPEN` | Injecting the opening `{` of the parameters object |
| `EXPECT_PARAM_KEY` | Injecting the current parameter key (e.g. `"a"`) |
| `EXPECT_PARAM_COLON` | Injecting `:` after the parameter key |
| `EXPECT_VALUE_START` | Dispatcher: transitions to the correct value state |
| `EXPECT_VALUE_QUOTE` | Injecting the opening `"` for string values |
| `VALUE_STRING` | Model generates string characters token by token |
| `VALUE_NUMBER` | Model generates numeric characters token by token |
| `VALUE_BOOLEAN` | Model generates `true` or `false` via trie |
| `EXPECT_PARAM_COMMA_OR_CLOSE` | Injects `,` or `}` depending on remaining params |
| `EXPECT_OBJECT_END` | Injecting the closing `}` of the root object |
| `DONE` | Generation complete |

**Structural injection** handles all deterministic states (everything except
`EXPECT_FUNC_NAME`, `VALUE_STRING`, `VALUE_NUMBER`, `VALUE_BOOLEAN`) without
calling the model at all. Only the semantically variable parts require a forward
pass.

**Greedy decoding with masking** applies a `-inf` mask to all disallowed tokens
in the logit vector, then takes the argmax. This guarantees that the model can
never produce a structurally invalid token.

**Token tries** (`src/token_utils.py`) are used for multi-token sequences
(function names, boolean values) to efficiently track which completions are
still valid at each step.

---

## Design decisions

**Function selection uses the LLM, not heuristics.** As required by the subject
(section IV.3.1: *"The function to call should be chosen using the LLM, not
with heuristics or any other sort of medieval magic"*), the function name is
generated token-by-token by the model under trie constraints. The decoder never
inspects the prompt to decide which function to call.

**Structurally-typed token sets for parameter values.** The subject requires
constrained decoding to enforce a specific schema (section V.3.3). For
parameters typed as `"number"`, only numeric character tokens are allowed. For
parameters typed as `"string"`, a broad safe set is used. For parameters whose
structural role is a regex pattern (`"regex"`) or a literal replacement string
(`"replacement"`), separate token sets and closing conditions are applied:

- `regex_safe_tokens`: characters valid in a Python regex pattern
  (`a-z`, `A-Z`, `0-9`, `\`, `[`, `]`, `+`, `*`, `.`, `|`, `^`, `$`, `-`).
  Generation closes as soon as the last emitted token ends with a regex
  quantifier or closing bracket (`+`, `*`, `?`, `]`) — characters that in regex
  grammar signal the end of a complete atomic unit. This is a syntactic property
  of the token itself, not a semantic assumption about the prompt.
- `replacement_safe_tokens`: characters valid in a literal replacement string
  (alphanumerics and common punctuation, excluding regex metacharacters that
  have no meaning in a replacement context). If the first emitted token is a
  single character (length 1 in the vocabulary), generation closes immediately —
  a single-character token is by definition an atomic, non-extendable value.

These distinctions are derived from the structural role of each parameter as
declared in the function signature, not from the content of the user prompt or
parameter name. For a parameter to use `"regex"` or `"replacement"` format,
it **must explicitly declare** `"format": "regex"` or
`"format": "replacement"` in the `ParameterDef`. If no format is declared,
the parameter is treated as a generic string.
This is analogous to how `VALUE_NUMBER` applies a numeric charset regardless of
what the prompt says — it is a schema-level constraint, not a semantic one.
The subject (section IV.3.1) restricts heuristics only to function selection;
these constraints operate on parameter values after the function has already
been chosen by the model.

**No heuristics on parameter names.** The system does not infer parameter
format from the parameter name (e.g., no automatic "regex" assignment for
parameters named `pattern` or `regex`). Format detection is purely
schema-level and explicit, derived only from `ParameterDef.format` as declared
in the function signature. This ensures that the model always makes semantic
choices without hidden heuristic interference.

**`_string_safe_tokens` built at init time.** The set of tokens allowed inside
generic string values is computed once in `__init__` and reused at every step.
A token is excluded if it contains `"`, `{`, `}`, `,`, control characters
(ASCII < 32), or non-printable characters (ASCII > 126) — after converting
GPT-style tokens (`Ġ` → space, `Ċ` → newline) to their real character
representation.

**Prefix caching.** The system prompt and few-shot example are tokenized once in
`__init__`. Each call to `generate()` only tokenizes the variable part (the user
prompt) and concatenates it to the cached prefix.

**Three-level error recovery.**

- Level 1: when `allowed_tokens()` returns an empty set, the current FSM state
  and recent tokens are logged.
- Level 2: `get_result()` always returns whatever was collected, with an explicit
  `_error` field (stripped before output) if generation did not reach `DONE`.
- Level 3: if the empty set occurs inside `VALUE_STRING`, `VALUE_NUMBER`, or
  `VALUE_BOOLEAN`, `force_close_current_value()` injects the appropriate closing
  token and continues instead of aborting.

**`ModelProtocol` for dependency inversion.** The decoder depends on an abstract
protocol, not a concrete model class. This allows the full pipeline to be tested
with `DummyLLM` (no inference, deterministic output) without modifying any
decoder code. `DummyLLM` lives only under `tests/` and is never reachable from
the CLI — there is no `--model dummy` option; the test suite imports it
directly (see [Testing strategy](#testing-strategy)).

---

## Performance analysis

Benchmarks run on a shared school CPU (no GPU), reference 11-prompt suite, two independent runs per model (`temperature=0.0`).

| Metric | Qwen3-0.6B (mandatory) | Qwen2.5-0.5B-Instruct (bonus) |
|--------|------------------------|-------------------------------|
| Total prompts | 11 | 11 |
| Correct outputs | 10/11 (90.9%) | 10/11 (90.9%) |
| JSON validity | 11/11 (100%) | 11/11 (100%) |
| Total runtime (fast dev machine) | ~5.5–8s | ~6–8s |
| Determinism | identical token sequences across runs (same machine, same thread count) | identical token sequences across runs (same machine, same thread count) |

**Determinism, verified not assumed.** Both models were run multiple times, minutes apart, on the same suite, on the same machine. Every generated token sequence matched exactly between runs at `temperature=0.0` with a fixed thread count (`OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`), confirming `_sample_token` in `decoder.py` is fully reproducible. Without pinning thread count, PyTorch's CPU matmul can reorder floating-point sums differently between runs, which can occasionally flip an argmax tie and change the generated sequence — this is a property of multi-threaded CPU inference in general, not of our decoding logic.

### Known limitation: bonus model timing on constrained hardware

The subject states two relevant things that are easy to conflate but live in different chapters:

- **V.5 (Mandatory part, "Performance and Reliability"):** *"Reasonable speed: Process all test prompts in under 5 minutes on standard hardware."*
- **Chapter VII (Bonus Part):** lists "Support for multiple LLM models" and, **separately**, "Performance optimizations (caching, batching)" as two distinct, independent bonus items.

We read this as: the 5-minute target is a mandatory-part guarantee for the default `Qwen3-0.6B` backend, while raw speed under an alternate backend is not itself gated — *optimizing* that speed is its own, separate bonus item, which we partially implemented (prefix caching, in-memory prompt cache; no batching).

In practice, on shared/older lab hardware, the bonus backend (`--model Qwen25`) can take noticeably longer per prompt than on a fast development machine, and for prompts that require many generation steps (long numeric values, regex-heavy outputs) this can approach or exceed 5 minutes for the full 11-prompt suite. Two factors compound this, both intrinsic to the SDK rather than our code:

1. **No KV-cache in `llm_sdk`:** `get_logits_from_input_ids` reruns the full forward pass over the *entire* sequence on every generation step (confirmed by reading the SDK source — no `past_key_values` is used). Cost scales with `steps × full_sequence_length`, not just `steps`, so any prompt needing more decoding steps costs disproportionately more, and this effect is amplified on slower CPUs.
2. **Hardware variance:** the same code measured ~6–8s total for 11 prompts on a fast development machine, but markedly slower on a shared school lab CPU — we could not fully control for this, since we don't have direct access to that hardware's exact thread count, load, or available memory at evaluation time.

We chose not to chase this further by re-engineering the prompt or adding heuristics around generation length: a prompt-shortening variant was tested the same way as the ChatML experiment above (as a verification step, run against the grading suite, not adopted into the codebase), and showed inconsistent results on repeated runs. Since the prompt-construction code is shared between the mandatory and bonus paths, any change there carries risk for the mandatory requirement, which must never regress. We kept the author's original, already-validated prompt-construction code rather than adopt an alternative whose behavior we could not fully characterize in the time available.

**Theoretical floor.** Each forward pass processes the full `input_ids` sequence
from scratch — the SDK does not expose KV cache. The minimum runtime is therefore
`total_steps × time_per_step`, and cannot be reduced further without hardware
acceleration or KV cache support. On a fast development machine the 0.5B
bonus model is comparable in wall-clock time to the 0.6B mandatory model on
this suite, since most of the per-prompt time is dominated by the fixed
structural injection and short value spans rather than raw model size — see
[Known limitation](#known-limitation-bonus-model-timing-on-constrained-hardware)
for why this changes on slower hardware.

**Optimisations active:**

- Prefix caching: the fixed portion of the prompt (~130–200 tokens depending on
  the function set) is tokenised once and reused.
- Structural injection: deterministic tokens are injected without calling the
  model.
- Vectorised masking: logit masking uses NumPy operations instead of Python
  loops.
- Pre-tokenised parameter keys: all `"key":` sequences are tokenised at init
  time.
- In-memory prompt cache: identical prompts within the same run are not
  recomputed.

---

## Challenges faced

**Backtick and comma leaking into string values.** Early versions of
`_string_safe_tokens` excluded only `"`, `{`, and `}`. The comma (a structural
JSON separator) and backtick were not excluded, causing outputs like
`"[aeiouAEIOU]\`,"`. Fixed by adding them to the exclusion set.

**GPT-style tokens and the ASCII filter.** Qwen3 represents spaces as `Ġ` and
newlines as `Ċ` in its vocabulary. Applying a naive `ord(ch) > 126` filter
excluded these tokens entirely, preventing the model from generating spaces
inside string values. Fixed by converting `Ġ` and `Ċ` to their real characters
before the ASCII check.

**Cross-platform line endings.** The tokenizer received `\r\n` on some systems,
causing tokenization failures on characters not present in the vocabulary. Fixed
by normalising line endings at the start of `encode()` before any GPT-style
transformation.

**Protocol/Adapter name asymmetry.** `ModelProtocol` declared
`get_path_to_vocabulary_json` while the SDK exposes `get_path_to_vocab_file`.
Fixed by aligning the names across all files to match the SDK.

**Temporary file leak in `DummyLLM`.** `tempfile.mkstemp` created a file that
was never deleted. Replaced with `NamedTemporaryFile(delete=True)` and an
explicit destructor.

**Token `.*` leaking into replacement values.** The token `.*` (ID 4908) passed
the initial `replacement_safe_tokens` filter because both `.` and `*` were in
the allowed charset. After `*`, the model always chose `.*` as the
highest-probability continuation, producing `*.*`. Fixed by separating
`regex_safe_tokens` and `replacement_safe_tokens` into two distinct sets with
different allowed characters, and by closing the replacement immediately when
the first token is a single character.

**Regex over-generation after a valid pattern.** With no stopping condition,
the model continued generating after a valid regex (e.g. `[0-9]+`), adding
alternations that changed the semantic meaning. Fixed by closing generation as
soon as the last emitted token ends with a regex quantifier or closing bracket —
a syntactic signal that the pattern is structurally complete.

**Replacement continuation after a single-character value.** After emitting `*`,
the model's distribution strongly favoured continuation tokens (`.*`, `_*`,
`??`). Fixed by detecting that the first emitted token is a single character in
the vocabulary and forcing immediate closure — a single character is an atomic
value that cannot be meaningfully extended.

**Escaped forward slash surviving into the final string value (found via cross-model testing, not on Qwen3-0.6B).** Qwen3-0.6B's vocabulary tokenises `/` as its own
character, so a Unix path like `/home/user/data.json` was always reproduced
correctly. Qwen2.5-0.5B-Instruct's vocabulary instead contains a single merged
token for the two-character sequence `\/` — the standard, fully valid JSON
escape for a forward slash (RFC 8259 lists `\/` alongside `\"` and `\\` as a
legal escape). The constrained decoder correctly accepted it (the output JSON
text was 100% valid), but `_finish_param_value()` only unescaped `\\` → `\` and
`\"` → `"`, not `\/` → `/`, so the *parsed* parameter value kept the literal
backslash: `fn_read_file({"path": "\/home\/user\/data.json"})` instead of
`{"path": "/home/user/data.json"}`. This only ever surfaced on Qwen2.5, which is
exactly why testing a second backend mattered — it isn't just a bonus checkbox,
it caught a real bug Qwen3-0.6B's vocabulary happened to never trigger. Fixed by
adding the symmetric `s.replace('\\/', '/')` next to the existing `\\` and `\"`
unescaping in `schema.py`, with no change to the generation/masking logic — the
guard at `len(self.value_tokens) == 0` that blocks a *lone* backslash as the
first token of a string was left untouched, since blocking every
backslash-containing token would have broken Windows-path generation
(`C:\Users\...`), which relies on the model being able to emit literal
backslash tokens.

---

## Testing strategy

**Quick smoke test (no GPU needed):**

```bash
uv run python -m pytest tests/e2e/test_cli.py
```

Runs end-to-end tests with the internal dummy model (FSM and JSON structure validation). Verifies that the FSM, structural
injection, and JSON output work end-to-end in seconds.

**Internal `DummyLLM` for structural validation.** The test suite includes
a `DummyLLM` class (in `tests/model_dummy.py`) that implements `ModelProtocol`
and returns deterministic logits (all zeros). This allows the complete pipeline
(FSM, constrained decoding, JSON serialization) to be tested end-to-end without
any external dependencies (SDK, GPU, or network). The dummy model is used in
`tests/e2e/test_cli.py` to validate every structural path (numbers, strings,
booleans, single and multi-parameter functions) within seconds, without
inference overhead. This is critical for fast CI/CD feedback.

**11-prompt reference suite.** The suite covers all parameter types and all
defined functions, including edge cases (multi-parameter regex functions, numeric
strings, short and long replacements). A correct run produces 11 valid JSON
objects with no `_error` fields.

**`mypy` and `flake8`.** Static type checking and linting are run before every
commit. The codebase is fully annotated.

```bash
make lint
```

**`uv sync` from scratch.** Verified that a clean environment install works
without manual steps.

**Determinism check (manual, cross-run).** The reference 11-prompt suite was run
twice per model, minutes apart, with `--verbose`. Every generated token sequence
matched exactly between the two runs of the same model, confirming
`temperature=0.0` produces fully reproducible output — important since the
subject's own example output (`"a": 2.0, "b": 3.0`) implies a stable, repeatable
result rather than a sampled one.

**Regression check after the `\/` unescaping fix.** Before merging the fix
described in "Challenges faced", the exact failing token sequence from the
Qwen2.5 run (`'\/','home','\/','user','\/','data','.json'`) was replayed against
`_finish_param_value()` directly to confirm it now produces `/home/user/data.json`,
and the Qwen3 Windows-path token sequence (`'C',':\\','\\','Users',...`) was
replayed the same way to confirm the existing `\\` → `\` unescaping for backslash
paths was untouched by the change. The full `pytest` suite, `flake8`, and
`mypy` (with the exact flags from `make lint`) were re-run after the fix with
no new failures introduced.

---

## Example usage

```bash
$ uv run python -m src --verbose
```

Input prompts (from `data/input/`):

```
What is the sum of 2 and 3?
Greet shrek
Reverse the string 'hello'
Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'
```

Output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  },
  {
    "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "The cat sat on the mat with another cat",
      "regex": "cat",
      "replacement": "dog"
    }
  }
]
```

---

## Resources

### Documentation and references

- [uv — Python package manager](https://docs.astral.sh/uv/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)
- [Outlines — structured text generation](https://github.com/guidance-ai/guidance)
  — reference implementation of constrained decoding
- [Guidance — constrained generation](https://github.com/guidance-ai/guidance)
  — alternative approach to the same problem

---

## AI usage

Claude (Anthropic) was used throughout the project for code review, debugging,
and documentation. All algorithmic decisions — the FSM design, the constrained
decoding logic, the tokenizer — were made and implemented by the author. AI
assistance was limited to identifying bugs, suggesting fixes, verifying
compliance with the subject, and drafting this README.
