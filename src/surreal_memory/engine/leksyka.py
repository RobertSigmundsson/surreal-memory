"""Pure-function tokenization for the W3 lexical gibberish signal.

Program `smem-recall-trzy-warstwy`, unit U2. Emulates, on the Python side and
with zero I/O, what the SurrealDB full-text index already does to `content`
at write time: `DEFINE ANALYZER smem_content TOKENIZERS blank, class FILTERS
lowercase, ascii` (`storage/surrealdb/schema.py:21`), backing `idx_neuron_
content_fts ON neuron FIELDS content FULLTEXT ANALYZER smem_content BM25`
(`storage/surrealdb/schema.py:63`). The `blank` tokenizer splits on
whitespace; `class` additionally splits at character-class boundaries
(letter/digit/punctuation), which is why an underscore inside a query like
`uruboros_kafka` produces two tokens, not one. The `ascii` filter folds
accented Latin letters to their ASCII base form.

Measured (82-query apparatus, program `smem-recall-trzy-warstwy` U1/U2
measurement pass): `content @@ 'a b'` is an AND over tokens, not an OR/phrase
match (`@@ 'nautilus'` = 274 rows, `@@ 'qwzlmnprt'` = 0, `@@ 'nautilus
qwzlmnprt'` = 0) -- so a single whole-phrase `@@` cannot serve as a
gibberish signal; the signal instead has to be "does ANY token of the query
appear ANYWHERE in the base", checked per-token. The rule "no token of
length >= 4 appears in the base at all" was measured to flag gibberish 5/5
while firing 0/49 on golden queries, 0/5 on `pl-naturalna`, 0/7 on
`time-liczba-data`, 0/6 on `indomain`, 4/5 on `no`, 1/5 on `en`. At a
threshold of length >= 3 the gibberish catch rate DROPS to 4/5 (tokens like
`zzz`/`dd` collide with hex fragments already present in the base) -- which
is why the measured default (`BrainConfig.refusal_observe_leksyka_min_token_
len`) is 4, not 3.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+")
_MAX_TOKENS = 32


def tokenizuj(query: str) -> list[str]:
    """Tokenize `query` the way `smem_content` tokenizes `content` at index time.

    Steps: lowercase -> fold `ł` -> `l` (NFKD alone does not decompose it, but
    the index's `ascii` filter folds it to `l`) -> NFKD-normalize and drop
    combining marks (folds other accented letters, e.g. `ą` -> `a`) -> split
    into maximal runs of ASCII letters or ASCII digits (emulates BOTH the
    `blank` tokenizer, which splits on whitespace, and the `class` tokenizer,
    which additionally splits at letter/digit/punctuation boundaries -- so an
    underscore or a letter/digit boundary is also a split point). Returns
    tokens in the order they occur; duplicates are NOT removed here (that is
    `tokeny_do_sprawdzenia`'s job).
    """
    lowered = query.lower().replace("ł", "l")  # "ł" -> "l"
    decomposed = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _TOKEN_RE.findall(stripped)


def tokeny_do_sprawdzenia(query: str, min_len: int) -> list[str]:
    """Tokens worth checking against the base: length >= `min_len`, de-duplicated
    (first occurrence kept, order preserved), capped at 32 -- a hard limit so a
    query with a hundred distinct long tokens cannot build an unbounded SQL
    `OR` chain in `any_neuron_matches_any_token`.
    """
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokenizuj(query):
        if len(tok) < min_len:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= _MAX_TOKENS:
            break
    return out
