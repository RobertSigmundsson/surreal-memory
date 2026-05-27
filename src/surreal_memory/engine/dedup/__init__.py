"""LLM-powered deduplication for anchor neurons.

3-tier cascade: SimHash -> Embedding cosine -> LLM judgment.
Each tier short-circuits on definitive answers.

OFF by default -- enable via DedupConfig(enabled=True).
"""

from surreal_memory.engine.dedup.config import DedupConfig
from surreal_memory.engine.dedup.pipeline import DedupPipeline, DedupResult

__all__ = ["DedupConfig", "DedupPipeline", "DedupResult"]
