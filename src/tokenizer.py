"""Custom tokenizer based exclusively on the JSON vocabulary.

Does not use the SDK encode/decode methods, but implements greedy
longest match tokenization and decoding with special character handling.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, List

from .token_utils import load_vocab

logger = logging.getLogger(__name__)


class _CharTrie:
    """Character trie for longest match token lookup in the vocabulary.

    Attributes:
        children: Mapping from character to the child trie node.
        token_id: The token ID if this node completes a valid token,
            otherwise ``None``.
    """

    #  CharTrie ---------------------------------------------------------------
    # "Naviga CARATTERI, non token id. Esiste solo per fare in modo che
    #  encode() trovi sempre il token più lungo possibile invece di
    #  spezzare il testo in tanti token corti — è un'ottimizzazione
    #  interna di encode(), non qualcosa con cui il decoder interagisce
    #  direttamente."

    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        """Initialise an empty trie node."""
        self.children: dict[str, "_CharTrie"] = {}
        self.token_id: Optional[int] = None

    def insert(self, token: str, token_id: int) -> None:
        """Insert a token and its ID into the trie.

        Args:
            token: The string representation of the token.
            token_id: The numeric ID assigned to this token in the
                vocabulary.
        """
        node = self
        for ch in token:
            if ch not in node.children:
                node.children[ch] = _CharTrie()
            node = node.children[ch]
        node.token_id = token_id


class CustomTokenizer:
    """Tokenizer that operates on the vocabulary loaded from a JSON file.

    Supports both direct vocabulary passing (to avoid double reading)
    and loading from a file.  Implements greedy longest match encoding
    and GPT style decoding with ``Ġ`` (space) and ``Ċ`` (newline)
    markers.

    Attributes:
        id_to_token: Mapping from token ID to string representation.
        token_to_id: Reverse mapping from string to token ID.
        _trie: Internal character trie used for efficient encoding.
    """

    #  CustomTokenizer:
    #   - encode(text) — greedy longest-match, spazio/newline diventano
    #     Ġ/Ċ, carattere sconosciuto → fallback UNK (mai eccezione).
    #   - decode(token_ids) — l'inverso, ripristinando spazio/newline."

    def __init__(self, vocab_source: str | Dict[int, str]) -> None:
        """Initialise the tokenizer.

        Args:
            vocab_source: Either a path to the vocabulary JSON file
                or an already loaded ``id -> token`` dictionary.
        """
        if isinstance(vocab_source, dict):
            self.id_to_token = vocab_source
        else:
            self.id_to_token = load_vocab(vocab_source)

        self.token_to_id = {v: k for k, v in self.id_to_token.items()}

        # Il _CharTrie viene costruito qui, usando i token del vocabolario
        self._trie = _CharTrie()
        for token, tid in self.token_to_id.items():
            self._trie.insert(token, tid)

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of token IDs using greedy.

        Space and newline characters are transformed into ``Ġ`` and
        ``Ċ``, similarly to GPT style tokenization. Characters not
        present in the vocabulary are handled with an UNK fallback
        token instead of raising an exception, to ensure robustness.

        Args:
            text: The string to tokenize.

        Returns:
            List of corresponding token IDs.
        """
        logger.debug("encode input (first 80 chars): %s", repr(text[:80]))

        #  Encode - Greedy longest-match:
        #  ad ogni posizione, scende nel _CharTrie
        #  finché può, prende il token più lungo trovato. Spazio e
        #  newline diventano Ġ/Ċ PRIMA di tokenizzare (stile GPT).
        #  Carattere sconosciuto → fallback UNK, mai eccezione: per
        #  questo è 'robusto' nella sua stessa docstring."

        text = text.replace(' ', 'Ġ').replace('\n', 'Ċ')
        ids: List[int] = []
        idx = 0
        n = len(text)
        unk_count = 0
        # Greedy longest-match: -----------------------------------------------
        # per ogni posizione, scende nel _CharTrie
        while idx < n:
            node = self._trie
            last_match_id = None
            last_match_end = None
            pos = idx
            while pos < n and text[pos] in node.children:
                node = node.children[text[pos]]
                pos += 1
                if node.token_id is not None:
                    last_match_id = node.token_id
                    last_match_end = pos
            if last_match_id is not None:
                assert last_match_end is not None
                ids.append(last_match_id)
                idx = last_match_end
            else:
                # Carattere non presente: fallback UNK ------------------------
                char = text[idx]
                if char in self.token_to_id:
                    ids.append(self.token_to_id[char])
                    idx += 1
                else:
                    unk_id = self.token_to_id.get('�', 0)
                    unk_count += 1
                    logger.warning(
                        "Unknown character '%s' "
                        "(pos %d, char code %d) -> UNK token %d",
                        char, idx, ord(char), unk_id,
                    )
                    ids.append(unk_id)
                    idx += 1

        logger.debug(
            "encode output: %d tokens (input_len=%d, unk_count=%d)",
            len(ids),
            n,
            unk_count
        )
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into a string.

        Handles GPT‑style tokens: ``Ġ`` is converted to space, ``Ċ``
        to newline.

        Args:
            token_ids: List of token IDs.

        Returns:
            The decoded string.
        """
        logger.debug("decode input: %d tokens", len(token_ids))
        #  decode() - Da IDs a STRINGA
        # "L'inverso: da id a stringa, ripristinando spazio/newline da
        #  Ġ/Ċ."
        parts = [self.id_to_token.get(tid, "") for tid in token_ids]
        result = []
        space_count = 0
        newline_count = 0
        for tid, part in zip(token_ids, parts):
            if part.startswith("Ġ"):
                result.append(" " + part[1:])
                space_count += 1
            elif part.startswith("Ċ"):
                result.append("\n" + part[1:])
                newline_count += 1
            else:
                result.append(part)
        text = "".join(result)
        logger.debug(
            "decode output: %d chars (space_tokens=%d, newline_tokens=%d)",
            len(text), space_count, newline_count
        )
        return text
