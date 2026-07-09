"""Regression guard for BUG-10 (W7.3): a single hardened ``_to_surreal_id``.

Before this fix, 12 SurrealDB storage mixins each defined their own
``_to_surreal_id`` — all pre-BUG-8 unhardened copies (``record_id.replace("-",
"_")``, three of them with a prefix-strip). Those fed raw f-string-inlined
sinks (``UPSERT typed_memory:{sid} CONTENT $data``, ``UPSERT project:{sid}
CONTENT $data``, ``UPSERT source:{sid} CONTENT $data``, plus assorted
``delete/merge/select(f"table:{sid}")`` calls) without folding characters
that break out of the record literal / statement.

This test asserts every one of the 12 mixins (plus ``store.py``) now imports
the *exact same function object* from ``surreal_memory.storage.surrealdb._ids``
— so the "single choke-point, no un-sanitised id reaches the engine" guarantee
is structurally true, not just behaviorally coincidental — and fuzzes that
function with breakout payloads plus the specific hostile ids named in BUG-10.
"""

from __future__ import annotations

import importlib

import pytest

from surreal_memory.storage.surrealdb._ids import _safe_brain_id, _to_surreal_id

# The 12 mixins that previously carried their own unhardened copy, plus the
# store module (which re-exports the hardened names for existing tests /
# callers that do `from ...store import _to_surreal_id`).
_MIXIN_MODULES = [
    "surreal_memory.storage.surrealdb.compression",
    "surreal_memory.storage.surrealdb.maturation",
    "surreal_memory.storage.surrealdb.review_schedules",
    "surreal_memory.storage.surrealdb.alerts",
    "surreal_memory.storage.surrealdb.typed_memory",
    "surreal_memory.storage.surrealdb.cognitive",
    "surreal_memory.storage.surrealdb.activity",
    "surreal_memory.storage.surrealdb.projects",
    "surreal_memory.storage.surrealdb.depth_priors",
    "surreal_memory.storage.surrealdb.keyword_entity",
    "surreal_memory.storage.surrealdb.versions",
    "surreal_memory.storage.surrealdb.sources",
]

_STORE_MODULE = "surreal_memory.storage.surrealdb.store"

# Breakout payloads spanning quote/brace/paren/semicolon/whitespace/backtick/
# comment/unicode classes.
_HOSTILE_PAYLOADS = [
    'x"',
    'x"})',
    'x"}) RETURN s //',
    'aaa"})-[:synapse]->{1,4}(t:neuron) RETURN s //',
    'zzz"}) RETURN (MATCH (a:neuron) RETURN a) AS leaked //',
    "a' OR 1=1 --",
    "id with spaces",
    "back`tick",
    "semi;colon",
    "star*glob",
    'fiber_id"} ; UPDATE typed_memory SET trust_score=99 RETURN {',
    "paren)close",
    "paren(open",
    "brace{open",
    "brace}close",
    "bracket[open",
    "bracket]close",
    "angle<tag>",
    "back\\slash",
    "tab\tnewline\n",
    "unicode_‮override",
]

# The specific hostile fiber_id / version_id / project / source breakout
# strings called out in BUG-10.
_NAMED_HOSTILE_IDS = {
    "fiber_id": 'abc"} ; UPDATE typed_memory SET trust_score=99 RETURN {',
    "version_id": 'v1"}) ; REMOVE TABLE brain_version; //',
    "project": 'proj"} ; UPDATE project SET owner="attacker" RETURN {',
    "source": 'src"} ; DELETE source; SELECT * FROM typed_memory WHERE "',
}

_ALLOWED_CHARSET = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def _all_modules_under_test() -> list[str]:
    return [*_MIXIN_MODULES, _STORE_MODULE]


class TestSingleSourceOfTruth:
    """Every mixin + store must resolve `_to_surreal_id` to the SAME object."""

    @pytest.mark.parametrize("module_path", _all_modules_under_test())
    def test_mixin_imports_the_shared_function_object(self, module_path):
        module = importlib.import_module(module_path)
        imported = module._to_surreal_id
        assert imported is _to_surreal_id, (
            f"{module_path}._to_surreal_id is not the same object as "
            f"surreal_memory.storage.surrealdb._ids._to_surreal_id — a local "
            f"copy has been reintroduced, breaking the single choke-point "
            f"guarantee (BUG-10)."
        )

    def test_store_reexports_safe_brain_id_identity(self):
        store = importlib.import_module(_STORE_MODULE)
        assert store._safe_brain_id is _safe_brain_id

    def test_ids_module_is_the_only_definition_site(self):
        """Belt-and-suspenders: source-scan every module under test and assert
        none of them re-defines `_to_surreal_id` locally (only imports it)."""
        import inspect

        for module_path in _all_modules_under_test():
            module = importlib.import_module(module_path)
            src = inspect.getsource(module)
            assert "def _to_surreal_id" not in src, (
                f"{module_path} still defines its own _to_surreal_id"
            )


class TestFuzzAcrossAllSources:
    """Fuzz `_to_surreal_id` as imported from every one of the 13 modules with
    ~20 breakout payloads; output must always be within [A-Za-z0-9_]."""

    @pytest.mark.parametrize("module_path", [*_all_modules_under_test(), "surreal_memory.storage.surrealdb._ids"])
    @pytest.mark.parametrize("payload", _HOSTILE_PAYLOADS)
    def test_output_always_charset_safe(self, module_path, payload):
        module = importlib.import_module(module_path)
        fn = module._to_surreal_id
        out = fn(payload)
        assert set(out) <= _ALLOWED_CHARSET, (
            f"{module_path}._to_surreal_id({payload!r}) -> {out!r} leaked a "
            f"non-charset character"
        )


class TestNamedHostileBreakoutStringsFoldToInert:
    """The specific hostile fiber_id/version_id/project/source ids named in
    BUG-10 must fold to something with zero SurQL-breakout characters,
    regardless of which of the 13 modules `_to_surreal_id` is imported from."""

    @pytest.mark.parametrize("module_path", _all_modules_under_test())
    @pytest.mark.parametrize("kind,payload", list(_NAMED_HOSTILE_IDS.items()))
    def test_named_hostile_id_is_inert(self, module_path, kind, payload):
        module = importlib.import_module(module_path)
        fn = module._to_surreal_id
        out = fn(payload)
        assert set(out) <= _ALLOWED_CHARSET, (
            f"[{kind}] {module_path}._to_surreal_id({payload!r}) -> {out!r} "
            f"still contains breakout characters"
        )
        for breakout_char in '"\'{}()[]<>;:/*\\`= \t\n':
            assert breakout_char not in out, (
                f"[{kind}] {module_path}: {breakout_char!r} survived "
                f"sanitisation of {payload!r} -> {out!r}"
            )
