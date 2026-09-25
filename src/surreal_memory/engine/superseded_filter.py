"""One superseded-fact filter for every recall path (MCP, HTTP, CLI, hook, ``smem q``).

A fact whose ``typed_memory.valid_until`` is set was replaced by a newer one
(``engine/supersession.py``). ``recall_api`` hard-filtered such fibers since v2.9.0, but the
paths that print ``ReflexPipeline.query`` output directly (``recall_like_cli`` behind
``smem recall`` and ``/v1/recall-cli``, the ``UserPromptSubmit`` hook, ``smem q``) did not, so a
revoked rule could still come back as the top memory. This module holds the predicate, the
escape hatch and the prose rebuild in one place, so every path has the same semantics.

Filtering the fiber list is not enough: the pipeline prose also lists the anchor neuron of a
matched fiber under "Related Information" (the top activations), so the rebuild excludes the
anchors of dropped fibers and any activated neuron that belongs to a superseded fact — stamped
``_superseded`` or found only in fibers with ``valid_until``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from surreal_memory.engine.activation import ActivationResult
    from surreal_memory.safety.encryption import MemoryEncryptor

logger = logging.getLogger(__name__)

DISABLE_ENV = "SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER"
# ``format_context`` lists at most this many activations under "Related Information".
_RELATED_TOP_N = 20
# Fibers looked up per activated neuron; a neuron in this many or more counts as shared (kept).
_FIBERS_PER_NEURON = 20


def superseded_filter_enabled() -> bool:
    """Whether valid_until-set (superseded) facts are hard-filtered from recall.

    Escape hatch: set ``SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER`` to a truthy value to keep
    superseded facts (diagnostics). Read at call time, so one process can toggle it.
    """
    raw = os.getenv(DISABLE_ENV, "").strip().lower()
    return raw not in ("1", "true", "yes", "on")


def is_excluded_by_validity(
    tm: Any, *, valid_at: datetime | None, include_superseded: bool
) -> bool:
    """The validity predicate of ``recall_api`` (point-in-time + the default hard filter).

    ``tm`` is the fiber's ``TypedMemory`` or ``None``; a fiber without one is never excluded.
    """
    if tm is None:
        return False
    if valid_at is not None:
        return not tm.is_valid_at(valid_at)
    return (
        isinstance(tm.valid_until, datetime)
        and not include_superseded
        and superseded_filter_enabled()
    )


def norm_id(value: str) -> str:
    """Compare-form of a record id: no table prefix, ``_`` and ``-`` folded together.

    Fibers keep ``neuron_ids`` with dashes while neuron record ids carry underscores; both
    sides of every comparison go through this function, so non-UUID ids stay consistent.
    """
    text = str(value)
    if ":" in text:
        text = text.split(":", 1)[1]
    return text.strip("`⟨⟩").replace("_", "-")


@dataclass(frozen=True)
class SupersededFilterOutcome:
    """What ``filter_superseded`` did. ``result`` is the SAME object when nothing changed."""

    result: Any
    excluded_fiber_ids: list[str] = field(default_factory=list)
    excluded_neuron_ids: list[str] = field(default_factory=list)
    context_rebuilt: bool = False
    enabled: bool = True


def encryptor_from_config(config: Any) -> MemoryEncryptor | None:
    """The memory encryptor when encryption is enabled in ``config``; ``None`` otherwise."""
    try:
        if not config.encryption.enabled:
            return None
        from pathlib import Path

        from surreal_memory.safety.encryption import MemoryEncryptor

        keys_dir_str = getattr(config.encryption, "keys_dir", "")
        keys_dir = Path(keys_dir_str) if keys_dir_str else (config.data_dir / "keys")
        return MemoryEncryptor(keys_dir=keys_dir)
    except Exception:
        logger.debug("encryptor unavailable", exc_info=True)
        return None


def _replace(result: Any, **fields: Any) -> Any:
    import dataclasses

    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.replace(result, **fields)
    replace = getattr(result, "_replace", None)
    if callable(replace):
        return replace(**fields)
    raise TypeError(f"cannot replace fields on {type(result).__name__}")


def _activations(result: Any, exclude: set[str]) -> dict[str, ActivationResult]:
    """Activations for the rebuild: the pipeline's full map, else co-activations; minus ``exclude``."""
    from surreal_memory.engine.activation import ActivationResult

    levels = (getattr(result, "metadata", None) or {}).get("activation_levels")
    acts: dict[str, ActivationResult] = {}
    if isinstance(levels, dict) and levels:
        for nid, level in levels.items():
            if norm_id(nid) in exclude:
                continue
            acts[nid] = ActivationResult(
                neuron_id=nid,
                activation_level=float(level),
                hop_distance=0,
                path=[nid],
                source_anchor=nid,
            )
        return acts
    for co in getattr(result, "co_activations", []) or []:
        for nid in co.neuron_ids:
            if norm_id(nid) in exclude:
                continue
            acts.setdefault(
                nid,
                ActivationResult(
                    neuron_id=nid,
                    activation_level=co.binding_strength,
                    hop_distance=0,
                    path=[nid],
                    source_anchor=nid,
                ),
            )
    return acts


async def _superseded_among(storage: Any, ids: list[str]) -> set[str]:
    """Normalised ids among ``ids`` that the prose must not list: stamped ``_superseded``, or found
    ONLY in superseded fibers (every fiber holding the neuron has ``typed_memory.valid_until``).

    A neuron in no fiber, in a valid fiber, or in ``_FIBERS_PER_NEURON`` fibers or more (the lookup
    is capped, the rest unknown) stays.
    """
    neurons = await storage.get_neurons_batch(ids)
    out = {
        norm_id(nid)
        for nid, neuron in neurons.items()
        if neuron is not None and (neuron.metadata or {}).get("_superseded") is True
    }
    rest = [nid for nid in ids if norm_id(nid) not in out]
    if not rest:
        return out
    fibers = await storage.find_fibers_batch(rest, limit_per_neuron=_FIBERS_PER_NEURON)
    if not fibers:
        return out
    typed = await storage.get_typed_memories_batch([f.id for f in fibers])
    members = [({norm_id(x) for x in f.neuron_ids}, f.id) for f in fibers]
    for nid in rest:
        key = norm_id(nid)
        holders = [fid for ids_in, fid in members if key in ids_in]
        if (
            holders
            and len(holders) < _FIBERS_PER_NEURON
            and all(
                is_excluded_by_validity(typed.get(fid), valid_at=None, include_superseded=False)
                for fid in holders
            )
        ):
            out.add(key)
    return out


async def superseded_neurons(result: Any, storage: Any) -> set[str]:
    """Normalised ids of the activated neurons the prose would list that belong to superseded facts.

    ``format_context`` lists the top ``_RELATED_TOP_N`` activations; leaving one out moves the next
    into that window, so the window is re-checked until it holds no superseded neuron.
    """
    levels = (getattr(result, "metadata", None) or {}).get("activation_levels")
    ids: list[str] = []
    if isinstance(levels, dict) and levels:
        ids = [nid for nid, _ in sorted(levels.items(), key=lambda kv: kv[1], reverse=True)]
    else:
        for co in getattr(result, "co_activations", []) or []:
            ids.extend(co.neuron_ids)
    ids = list(dict.fromkeys(ids))
    excluded: set[str] = set()
    checked: set[str] = set()
    while True:
        window = [nid for nid in ids if norm_id(nid) not in excluded][:_RELATED_TOP_N]
        fresh = [nid for nid in window if nid not in checked]
        if not fresh:
            return excluded
        checked.update(fresh)
        excluded |= await _superseded_among(storage, fresh)


async def rebuild_context(
    result: Any,
    fiber_ids: list[str],
    storage: Any,
    *,
    exclude_neuron_ids: set[str],
    max_tokens: int,
    brain_id: str,
    clean_for_prompt: bool = False,
    encryptor: MemoryEncryptor | None = None,
) -> Any:
    """Rebuild ``result.context`` from ``fiber_ids`` without the excluded neurons.

    ``exclude_neuron_ids`` holds ``norm_id`` forms. With no surviving fiber and no activation
    left the prose is EMPTY — the pre-filter prose must never come back.
    """
    from surreal_memory.engine.retrieval_context import format_context

    fibers_ordered: list[Any] = []
    for fid in fiber_ids:
        fiber = await storage.get_fiber(fid)
        if fiber:
            fibers_ordered.append(fiber)
    acts = _activations(result, exclude_neuron_ids)
    if not fibers_ordered and not acts:
        return _replace(result, context="")
    new_ctx, _ = await format_context(
        storage=storage,
        activations=acts,
        fibers=fibers_ordered,
        max_tokens=max_tokens,
        encryptor=encryptor,
        brain_id=brain_id,
        clean_for_prompt=clean_for_prompt,
    )
    return _replace(result, context=new_ctx or "")


async def excluded_anchor_ids(storage: Any, fiber_ids: list[str]) -> set[str]:
    """Normalised anchor neuron ids of ``fiber_ids`` (fibers that no longer exist are skipped)."""
    out: set[str] = set()
    for fid in fiber_ids:
        fiber = await storage.get_fiber(fid)
        if fiber is not None and fiber.anchor_neuron_id:
            out.add(norm_id(fiber.anchor_neuron_id))
    return out


async def filter_superseded(
    result: Any,
    storage: Any,
    *,
    max_tokens: int,
    brain_id: str,
    clean_for_prompt: bool = False,
    encryptor: MemoryEncryptor | None = None,
    config: Any = None,
) -> SupersededFilterOutcome:
    """Drop fibers whose ``typed_memory.valid_until`` is set, and their prose.

    Only ``valid_until`` excludes here (no expiry, trust or tier) — a path that never had those
    filters keeps its ranking for every fiber that is not superseded. Fibers without a
    ``typed_memory`` row stay. A matched list without a superseded fiber still gets its prose
    rebuilt when a top activation is stamped ``_superseded``. Nothing to exclude (or the escape
    hatch set) returns the SAME ``result`` object, so the prose is byte-identical. ``config`` (a
    ``UnifiedConfig``) supplies the encryptor lazily, only when a rebuild is needed.
    """
    if not superseded_filter_enabled():
        return SupersededFilterOutcome(result, enabled=False)
    matched = getattr(result, "fibers_matched", None)
    if not isinstance(matched, list) or not matched:
        return SupersededFilterOutcome(result)
    typed = await storage.get_typed_memories_batch(matched)
    excluded = [
        fid
        for fid in matched
        if is_excluded_by_validity(typed.get(fid), valid_at=None, include_superseded=False)
    ]
    # A superseded fact can reach the prose without its fiber in the matched list: its anchor or
    # another of its neurons is an activated neuron under "Related Information" (measured on a
    # copy of the brain: the anchor in 1 query of 136, a neuron of the same fiber in 1 more).
    stamped = await superseded_neurons(result, storage)
    if not excluded and not stamped:
        return SupersededFilterOutcome(result)
    kept = [fid for fid in matched if fid not in set(excluded)]
    exclude_neurons = await excluded_anchor_ids(storage, excluded) | stamped
    if encryptor is None and config is not None:
        encryptor = encryptor_from_config(config)
    filtered = _replace(result, fibers_matched=kept)
    rebuilt = await rebuild_context(
        filtered,
        kept,
        storage,
        exclude_neuron_ids=exclude_neurons,
        max_tokens=max_tokens,
        brain_id=brain_id,
        clean_for_prompt=clean_for_prompt,
        encryptor=encryptor,
    )
    return SupersededFilterOutcome(
        rebuilt,
        excluded_fiber_ids=excluded,
        excluded_neuron_ids=sorted(exclude_neurons),
        context_rebuilt=True,
    )
