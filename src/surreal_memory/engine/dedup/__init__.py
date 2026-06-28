"""LLM-powered deduplication for anchor neurons.

3-tier cascade: SimHash -> Embedding cosine -> LLM judgment.
Each tier short-circuits on definitive answers.

OFF by default -- enable via DedupConfig(enabled=True).
"""

from typing import Any

from surreal_memory.engine.dedup.config import DedupConfig
from surreal_memory.engine.dedup.pipeline import DedupPipeline, DedupResult

__all__ = ["DedupConfig", "DedupPipeline", "DedupResult", "build_dedup_pipeline"]


def build_dedup_pipeline(config: Any, storage: Any) -> "DedupPipeline | None":
    """Build a DedupPipeline from unified config ``[dedup]`` settings, or None
    if dedup is disabled/misconfigured.

    Shared by every encode() caller so anchor-dedup does NOT depend on which
    path performs the write. Previously this build was inline only in
    remember_handler; the stop-hook and other hooks constructed encoders
    WITHOUT a dedup pipeline, so identical anchors (e.g. repeated
    "Session activity" summaries) accumulated unchecked.
    """
    try:
        dedup_settings = config.dedup
        if not (isinstance(dedup_settings.enabled, bool) and dedup_settings.enabled):
            return None

        dedup_cfg = DedupConfig(
            enabled=True,
            simhash_threshold=int(dedup_settings.simhash_threshold),
            embedding_threshold=float(dedup_settings.embedding_threshold),
            embedding_ambiguous_low=float(dedup_settings.embedding_ambiguous_low),
            llm_enabled=bool(dedup_settings.llm_enabled),
            llm_provider=str(dedup_settings.llm_provider),
            llm_model=str(dedup_settings.llm_model),
            llm_max_pairs_per_encode=int(dedup_settings.llm_max_pairs_per_encode),
            merge_strategy=str(dedup_settings.merge_strategy),
            max_candidates=int(dedup_settings.max_candidates),
        )

        llm_judge = None
        if dedup_cfg.llm_enabled and dedup_cfg.llm_provider != "none":
            from surreal_memory.engine.dedup.llm_judge import create_judge

            llm_judge = create_judge(dedup_cfg.llm_provider, dedup_cfg.llm_model)

        return DedupPipeline(config=dedup_cfg, storage=storage, llm_judge=llm_judge)
    except (AttributeError, TypeError, ValueError):
        return None
