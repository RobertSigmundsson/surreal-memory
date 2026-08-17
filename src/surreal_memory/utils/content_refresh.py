"""Shared helper for keeping content-derived fields in sync with a content change.

Lives in ``utils/`` (not ``engine/`` or ``mcp/``) because every layer that changes
a neuron's content — the MCP edit tool, instruction refinement, and the
compression engine — needs it, and ``utils/`` is the one layer none of them
import back from (see ``simhash``, which sits here for the same reason).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from surreal_memory.core.brain import Brain
    from surreal_memory.core.neuron import Neuron
    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


async def contents_refreshed(
    storage: NeuralStorage,
    changes: list[tuple[Neuron, str]],
    *,
    brain: Brain | None = None,
) -> list[Neuron]:
    """Refresh many neurons' content-derived fields in ONE provider round-trip.

    ``content_hash`` is a pure function of the content, so it is recomputed for
    every neuron unconditionally — keeping the old fingerprint would feed
    near-duplicate detection the SimHash of text that no longer exists.

    The embedding is recomputed only for neurons that already carry one
    (``metadata["_embedding"]``, surfaced by the storage read). ``update_neuron``
    writes whatever vector that key holds, so without a refresh a content change
    actively re-saves the OLD vector against the NEW text: the memory stays
    retrievable by what it used to say, and ``reindex --missing-only`` cannot
    repair it because the field is never empty.

    Batching matters because the compression engine calls this from loops over
    every neuron of a fiber, under a time budget, in a path that already counts
    database round-trips as its dominant cost. One brain lookup and one
    ``embed_batch`` cover the whole batch, mirroring how the write path embeds on
    create (``encoder.py``) rather than paying a round-trip per neuron.

    Re-embedding goes through the same provider path and bounded wait the write
    path uses. If the provider is unavailable the change still succeeds, but the
    stale vectors are reported with a warning naming the repair command instead
    of being rewritten silently.

    ``brain`` lets a caller that loops over many fibers (``compress_all``) fetch
    the brain once and reuse it, instead of paying one lookup per fiber for a
    value that cannot change mid-pass. When omitted, it is fetched here.
    """
    from dataclasses import replace as dc_replace

    from surreal_memory.utils.simhash import simhash

    updated: list[Neuron] = []
    metas: list[dict[str, object]] = []
    for neuron, new_content in changes:
        # dc_replace keeps this exact dict object, so mutating it below also
        # updates the returned neuron's metadata.
        meta = dict(neuron.metadata)
        metas.append(meta)
        updated.append(
            dc_replace(
                neuron, content=new_content, content_hash=simhash(new_content), metadata=meta
            )
        )

    targets = [i for i, meta in enumerate(metas) if "_embedding" in meta]
    if not targets:
        return updated

    scope = f"neuron {updated[targets[0]].id}" if len(targets) == 1 else f"{len(targets)} neurons"
    try:
        import asyncio

        from surreal_memory.core.brain import BrainConfig
        from surreal_memory.engine.encoder import _inline_embed_timeout
        from surreal_memory.engine.semantic_discovery import _create_provider

        if brain is None:
            brain = await storage.get_brain(storage.brain_id or "")
        provider = _create_provider(
            brain.config if brain else BrainConfig(), task_type="RETRIEVAL_DOCUMENT"
        )
        # Embed each DISTINCT text once and fan its vector out. Identical text
        # must map to the identical vector anyway, and a GRAPH_ONLY pass hands
        # this helper one placeholder string per neuron — without the dedup
        # that is N copies of the same text in one provider payload.
        texts = [updated[i].embedding_text() for i in targets]
        unique_index: dict[str, int] = {}
        unique_texts: list[str] = []
        for text in texts:
            if text not in unique_index:
                unique_index[text] = len(unique_texts)
                unique_texts.append(text)
        embed = provider.embed_batch(unique_texts)
        budget = _inline_embed_timeout()
        vectors = await (asyncio.wait_for(embed, timeout=budget) if budget else embed)
        # Check the count before assigning any of it. A provider that returns a
        # short list would otherwise leave the tail neurons holding their OLD
        # vectors with no error and no warning — a silent stale write, which is
        # the exact failure this helper exists to remove.
        if len(vectors) != len(unique_texts):
            raise ValueError(
                f"embedding provider returned {len(vectors)} vectors for {len(unique_texts)} texts"
            )
        for i, text in zip(targets, texts, strict=True):
            metas[i]["_embedding"] = list(vectors[unique_index[text]])
    except TimeoutError:
        logger.warning(
            "Content of %s changed but re-embedding exceeded its time budget — "
            "the stored vectors still describe the old content; "
            "run `smem reindex --all` to repair them.",
            scope,
        )
    except Exception:
        logger.warning(
            "Content of %s changed but the embeddings could not be recomputed — "
            "the stored vectors still describe the old content; "
            "run `smem reindex --all` to repair them.",
            scope,
            exc_info=True,
        )
    return updated


async def content_refreshed(
    storage: NeuralStorage,
    neuron: Neuron,
    new_content: str,
    *,
    brain: Brain | None = None,
) -> Neuron:
    """Return *neuron* with its content replaced and the derived fields refreshed.

    Single-neuron form of :func:`contents_refreshed` — see there for why the hash
    is always recomputed and the vector only when one already exists.
    """
    return (await contents_refreshed(storage, [(neuron, new_content)], brain=brain))[0]
