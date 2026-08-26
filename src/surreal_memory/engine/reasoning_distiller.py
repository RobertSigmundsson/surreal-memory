"""Reasoning distiller: staged reasoning_traces -> ReasoningBank patterns.

Heuristic (no-LLM) distillation, run inside consolidation (strategy
``LEARN_REASONING``, SUMMARIZE tier, after ``PROCESS_REASONING_TRACES`` ingest):

  batch unprocessed traces per model
    -> segment each thinking into ~12 closed-vocabulary reasoning "moves"
    -> classify a category (bge-m3 embedding vs seed centroids; keyword fallback)
    -> cluster within (model, category) by cosine (embeddings) or move-set Jaccard
    -> for each cluster >= min_cluster_support: build a ReasoningBank pattern
       (title / description / strategy / confidence / frequency) and materialize
       it as a fiber (_reasoning_pattern) + CONCEPT neuron + EFFECTIVE_FOR synapse
    -> mark traces processed; prune + cap the staging table.

Fully fail-soft: with the embedding provider DOWN, classification falls back to
keywords and clustering to move-set Jaccard, so distillation still produces
patterns (never raises on a missing provider).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Direction, Synapse, SynapseType
from surreal_memory.engine.clustering import UnionFind
from surreal_memory.engine.reasoning_naming import _is_loopback, build_namer
from surreal_memory.engine.reasoning_progress import PHASE_DISTILLING, MiningProgress

if TYPE_CHECKING:
    from surreal_memory.engine.embedding.provider import EmbeddingProvider
    from surreal_memory.engine.reasoning_naming import PatternNamer
    from surreal_memory.engine.reasoning_progress import ProgressCallback
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

logger = logging.getLogger(__name__)

_CLUSTER_COSINE = 0.75  # fallback only; reasoning_training.cluster_cosine wins
_CATEGORY_COS_THRESHOLD = 0.35
_MOVE_JACCARD = 0.6
_CLASSIFY_CHARS = 500
# Moves needed before a chain counts as a pattern's identity (see merge_key).
_MIN_CHAIN_FOR_IDENTITY = 2
_BATCH_PER_MODEL = 200
# Ceiling on existing pattern fibers fetched for dedup/existing-count/coverage.
# Raised from 5000 for full-corpus mining across many models (u008).
_PATTERN_FETCH_LIMIT = 20_000

# ── Reasoning moves (closed vocabulary; regex discourse markers) ──────────────
#
# Calibrated 2026-08-26 against 35 733 real traces (per-move base rates and the
# quoted corpus evidence behind every phrase below live in the audit trail).
# The rule the old vocabulary broke: a marker must name the reasoning MOVE, not
# the discourse it happens to sit in. Three examples of what that cost:
#   * "i need to" alone fired on 45.8% of all traces and carried essentially the
#     whole restate-goal rate (46.0%). "I need to read the wiki" states the next
#     action, not the goal — it belongs to plan-steps if anywhere.
#   * bare "actually" fired on 31.6% and carried backtrack (34.4%). Most of those
#     are emphasis, not a reversal.
#   * bare "boundary" fired 470 times inside check-edge-cases, mostly on "scope
#     boundary" and similar — nothing to do with a boundary VALUE.
# A move that fires on nearly every trace cannot distinguish one strategy from
# another, which is how 218 of 295 pattern titles ended up containing
# restate-goal. Precision first; the measured rate is reported, not forced.
_REASONING_MOVES: dict[str, re.Pattern[str]] = {
    "restate-goal": re.compile(
        r"(?i)(\bthe goal (is|here is|was)\b|\bthe task (is|here is|was)\b|\bobjective is\b"
        r"|\bthe (user|robert) (wants|asked|is asking|said)\b"
        r"|\bwhat (he|she|they) (wants|asked|is asking)\b"
        r"|\bwhat (i'?m|we'?re) (trying|being asked) to\b"
        r"|\bthe (ask|request|requirement|instruction) is\b|\bmy job (is|here is)\b"
        r"|\bso,? (the|what) (goal|task|ask|question|requirement)\b"
        r"|\brestat\w+ the (goal|task)\b|\bthe point (is|of this is)\b"
        r"|\bwhat needs to (happen|be done)\b|\bwhat'?s being asked\b"
        r"|\bthe question is\b|\bboils down to\b)"
    ),
    "decompose": re.compile(
        r"(?i)(\bbreak (this|it|them|that) (down|into)\b|\bbreak(ing)? down\b"
        r"|\bdecompos(e|es|ed|ing)\b|\bsub-?problems?\b|\bthe steps are\b"
        r"|\bsplit (this|it|them|that) into\b|\bone (piece|part|step) at a time\b"
        r"|\bin (two|three|four|five) (steps|parts|stages|phases)\b"
        r"|\b(two|three|four|five) (separate )?(parts|pieces|stages|phases)\b"
        r"|\bpiece by piece\b|\bsmaller (pieces|chunks|steps)\b)"
    ),
    "hypothesize": re.compile(
        r"(?i)\b(hypothes\w*|i suspect|maybe|might be|could be|perhaps|likely because)\b"
    ),
    "gather-evidence": re.compile(
        r"(?i)\b(let me (check|look|read|grep)|looking at|the evidence|i see that|the code shows|confirmed that)\b"
    ),
    # Only verbs that name the ACT of verifying. Dropped: "make sure"/"ensure
    # that" (an imperative about care, not an act), and "confirmed"/"confirms"
    # — those report a RESULT, and "confirmed that" already belongs to
    # gather-evidence. Keeping them pushed this move to 43.5%, outside the band.
    "verify": re.compile(
        r"(?i)\b(verif(y|ying|ied)|confirm|double-?check|cross-?check"
        r"|sanity[- ]check|validate)\b"
    ),
    "test-first": re.compile(
        r"(?i)(\bwrit(e|ing|ten) an? (failing )?test\b|\btest[- ]first\b|\bfailing test\b"
        r"|\bred test\b|\btdd\b|\badd(ing|ed)? an? test\b|\bnegative control\b"
        r"|\breproduc\w+ (test|case)\b|\bregression test\b|\btest that (fails|would fail)\b"
        r"|\bprove it fails\b|\bmust fail on\b|\btest (first|before)\b|\bgolden (test|set|file)\b)"
    ),
    "check-edge-cases": re.compile(
        r"(?i)(\bedge cases?\b|\bcorner cases?\b|\bboundary (case|condition)\b"
        r"|\boff[- ]by[- ]one\b|\bempty (list|input|string|set|dict|result|response|file|output)\b"
        r"|\bnull case\b|\bnone case\b|\bwhat happens (if|when)\b|\bwhat if\b"
        r"|\bdegenerate case\b|\bzero[- ]length\b|\b(fails|breaks|blows up) (when|if)\b"
        r"|\bfirst (run|call|time)\b|\bmissing (file|key|field|value)\b|\brace condition\b)"
    ),
    "backtrack": re.compile(
        r"(?i)(\bwait[,.—:-]|\bhold on\b|\blet me reconsider\b|\bscratch that\b"
        r"|\bon second thought\b|\brethink\w*\b|\bstep(ping)? back\b|\bbut actually\b"
        r"|\bactually,? (no|wait|that|the|it|i)\b|\bthat'?s not right\b"
        r"|\bno[,—-] (wait|actually|that)\b"
        r"|\blet me re-?(examine|check|look|read|visit|assess|think)\b|\bon reflection\b"
        r"|\bhmm,? (but|wait|actually)\b|\breconsider\w*\b|\brevisit(ing)?\b"
        r"|\bchange[d]? my mind\b|\bbacktrack\w*\b)"
    ),
    "compare-alternatives": re.compile(
        r"(?i)\b(option [a-z0-9]|alternative|versus|vs\.?|instead of|trade-?off|on the other hand|compared to)\b"
    ),
    "plan-steps": re.compile(
        r"(?i)\b(my plan|the approach|step \d|next,? i|i'?ll (start|do|then)|let me first)\b"
    ),
    # Bare "correction" was 456 of this move's 516 hits and is almost always a
    # noun about someone else's fix ("Correction A:", "without the correction"),
    # not the author admitting an error.
    "self-correct": re.compile(
        r"(?i)(\bi was wrong\b|\bthat'?s incorrect\b|\bmy (mistake|error)\b|\bgot it wrong\b"
        r"|\boops\b|\bi mis(read|understood|counted|took|stated)\b|\bcorrecting myself\b"
        r"|\bi'?d claimed\b|\bmy (earlier|previous) (claim|assumption|reading|statement) was wrong\b"
        r"|\bi owe (you|him) a correction\b)"
    ),
    "summarize-decision": re.compile(
        r"(?i)\b(in summary|to summarize|conclusion|so i'?ll|decided to|the decision|therefore)\b"
    ),
}

# ── Category classification: seed descriptions (embeddings) + keyword fallback ─
_CATEGORY_SEEDS: dict[str, str] = {
    "debugging": "debugging errors, root cause analysis, stack traces, fixing bugs and failures",
    "planning": "planning steps, breaking down a task, deciding the approach and order of actions",
    "implementation": "implementing code, writing functions, adding a feature, coding the solution",
    "refactoring": "refactoring, cleaning up and restructuring code, improving readability, no behavior change",
    "research": "researching, reading documentation, exploring the codebase, understanding how something works",
    "verification": "verifying and testing, confirming correctness, checking outputs and edge cases",
    "architecture": "architecture and system design, module boundaries, data flow, design decisions",
    "data-analysis": "analyzing data, computing statistics, aggregating metrics, interpreting results",
}
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "debugging": (
        "bug",
        "error",
        "exception",
        "stack trace",
        "root cause",
        "traceback",
        "crash",
        "failing",
        "debug",
    ),
    "refactoring": (
        "refactor",
        "clean up",
        "rename",
        "restructure",
        "simplify",
        "extract",
        "dead code",
        "tidy up",
    ),
    "verification": (
        "verify",
        "test",
        "confirm",
        "validate",
        "assert",
        "make sure",
        "edge case",
        "double-check",
    ),
    "architecture": (
        "architecture",
        "design",
        "module",
        "boundary",
        "data flow",
        "interface",
        "layer",
        "component",
    ),
    "data-analysis": (
        "analyze",
        "statistics",
        "metric",
        "aggregate",
        "distribution",
        "compute",
        "measure",
    ),
    "research": (
        "read",
        "documentation",
        "docs",
        "explore",
        "investigate",
        "grep",
        "find out",
        "look up",
    ),
    "planning": ("plan", "approach", "steps", "strategy", "break down", "sequence", "outline"),
    "implementation": (
        "implement",
        "write the",
        "add a",
        "function",
        "build",
        "create the",
        "feature",
        "method",
    ),
}
# Keyword-fallback precedence (specific -> generic).
_CATEGORY_ORDER: tuple[str, ...] = (
    "debugging",
    "refactoring",
    "verification",
    "architecture",
    "data-analysis",
    "research",
    "planning",
    "implementation",
)

_OTHER = "other"


def _has_keyword(content_lower: str, keyword: str) -> bool:
    """Whole-word (single token) / substring (phrase) keyword match."""
    if " " in keyword or "-" in keyword:
        return keyword in content_lower
    return re.search(rf"\b{re.escape(keyword)}\b", content_lower) is not None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def segment_moves(text: str) -> list[str]:
    """Return the reasoning moves in *text* as a TEMPORAL sequence.

    Moves are ordered by where their marker actually occurs in the text, and
    an immediately repeated move is collapsed into one step (a trace that says
    "verify" three times in a row made one verification move, not three).
    A move that recurs LATER, after a different move, is kept: coming back to
    it is part of the shape of the reasoning.

    This used to iterate ``_REASONING_MOVES`` and return every move whose
    pattern matched anywhere — a SET of present moves, emitted in fixed
    vocabulary order. Downstream rendered that as "a -> b -> c", so every
    pattern claimed a chain while nothing in the pipeline had ever measured
    order. Ties (two markers at the same offset) fall back to vocabulary order
    so the sequence stays deterministic.
    """
    if not text:
        return []
    hits: list[tuple[int, int, str]] = []
    for rank, (move, pattern) in enumerate(_REASONING_MOVES.items()):
        hits.extend((match.start(), rank, move) for match in pattern.finditer(text))
    hits.sort()
    sequence: list[str] = []
    for _, _, move in hits:
        if not sequence or sequence[-1] != move:
            sequence.append(move)
    return sequence


def _classify_by_vector(vec: list[float], seeds: dict[str, list[float]]) -> str:
    best_cat, best_sim = _OTHER, _CATEGORY_COS_THRESHOLD
    for cat, cvec in seeds.items():
        sim = _cosine(vec, cvec)
        if sim >= best_sim:
            best_sim, best_cat = sim, cat
    return best_cat


def _classify_by_keywords(text: str, categories: tuple[str, ...]) -> str:
    low = text.lower()
    for cat in _CATEGORY_ORDER:
        if cat not in categories:
            continue
        if any(_has_keyword(low, kw) for kw in _CATEGORY_KEYWORDS.get(cat, ())):
            return cat
    return _OTHER


def _lcs_two(a: list[str], b: list[str]) -> list[str]:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    out: list[str] = []
    i = j = 0
    while i < m and j < n:
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def _lcs_all(seqs: list[list[str]]) -> list[str]:
    seqs = [s for s in seqs if s]
    if not seqs:
        return []
    acc = seqs[0]
    for s in seqs[1:]:
        acc = _lcs_two(acc, s)
        if not acc:
            break
    return acc


def _medoid_index(vectors: list[list[float]]) -> int:
    best_i, best_score = 0, -2.0
    for i in range(len(vectors)):
        score = sum(_cosine(vectors[i], vectors[j]) for j in range(len(vectors)) if j != i)
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def _cluster(
    vectors: list[list[float]] | None,
    moves: list[list[str]],
    cosine_threshold: float = _CLUSTER_COSINE,
) -> list[list[int]]:
    """Cluster items by cosine (vectors) or move-set Jaccard (fallback).

    ``cosine_threshold`` belongs to the configured embedder, not to this
    module: two models embedding the same pair of traces disagree on the
    absolute cosine, so a value tuned for one silently clusters nothing under
    another.
    """
    n = len(moves)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if vectors is not None:
                if _cosine(vectors[i], vectors[j]) >= cosine_threshold:
                    uf.union(i, j)
            else:
                a, b = set(moves[i]), set(moves[j])
                union = a | b
                if union and len(a & b) / len(union) >= _MOVE_JACCARD:
                    uf.union(i, j)
    return list(uf.groups().values())


def _move_idf(moves_per_trace: list[list[str]]) -> dict[str, float]:
    """Inverse document frequency of each move across a batch of traces.

    ``log(1 + N/df)`` rather than ``log(N/df)``: a move every trace in the batch
    makes would otherwise score exactly zero, and a small batch where all traces
    share their moves would produce no title at all. Smoothed, ubiquity is
    heavily discounted but still ordered.
    """
    n = len(moves_per_trace)
    if not n:
        return {}
    df = Counter(m for moves in moves_per_trace for m in set(moves))
    return {move: math.log(1.0 + n / count) for move, count in df.items()}


def _build_pattern(
    cluster_traces: list[dict[str, Any]],
    cluster_vectors: list[list[float]] | None,
    cluster_moves: list[list[str]],
    model: str,
    category: str,
    traces_in_category: int,
    *,
    move_idf: dict[str, float] | None = None,
) -> dict[str, Any]:
    size = len(cluster_traces)
    if cluster_vectors:
        mi = _medoid_index(cluster_vectors)
    else:
        mi = max(range(size), key=lambda i: len(str(cluster_traces[i].get("content", ""))))
    medoid_content = str(cluster_traces[mi].get("content", ""))

    # Count each move ONCE per trace: the title ranks "how many traces of this
    # cluster made this move", which is the document frequency, not how often a
    # marker fires inside one trace. segment_moves now returns a sequence that
    # may revisit a move, so the raw Counter would let a single verbose trace
    # decide the title.
    move_counts = Counter(m for moves in cluster_moves for m in dict.fromkeys(moves))
    # Weight by IDF over the batch this cluster came from. Raw frequency lets
    # the commonest move win every title, which is how "restate-goal" reached
    # 218 of 295 titles while saying nothing about what distinguishes one
    # strategy from another. A move that half the batch makes carries half the
    # information of one only this cluster makes. Ties break on the move name so
    # two runs over the same batch title a pattern the same way.
    idf = move_idf or {}
    top_moves = [
        m
        for m, _ in sorted(
            move_counts.items(), key=lambda kv: (-(kv[1] * idf.get(kv[0], 1.0)), kv[0])
        )[:3]
    ]
    title = f"{category}: " + ", ".join(top_moves) if top_moves else category

    lcs = _lcs_all(cluster_moves)
    if lcs:
        moves_line = "Moves: " + " -> ".join(lcs)
    elif top_moves:
        # No move ORDER is shared across the cluster. Rendering the frequency
        # top-3 with arrows anyway made "no common chain" byte-identical to a
        # measured chain, so the reader could not tell them apart. Say which
        # one this is.
        moves_line = "Moves (unordered): " + ", ".join(top_moves)
    else:
        moves_line = "Moves: (none detected)"
    strategy = f"{moves_line}\n{medoid_content[:400]}"[:600]

    confidence = min(1.0, size / traces_in_category) if traces_in_category else 0.0
    # Signature keys on the cluster's exact trace set (not the display title) so
    # two distinct clusters that share the same top-moves title never collide.
    trace_key = ",".join(sorted(str(t.get("trace_hash", "")) for t in cluster_traces))
    signature = hashlib.sha256(f"{model}:{category}:{trace_key}".encode()).hexdigest()
    return {
        "model": model,
        "category": category,
        "title": title,
        "description": medoid_content[:200],
        "strategy": strategy,
        "confidence": round(confidence, 4),
        "frequency": size,
        "signature": signature,
        # The measured chain, kept structured so the merge gate can key on it
        # without re-parsing the rendered strategy line.
        "chain": list(lcs),
    }


def merge_key(model: str, category: str, chain: Sequence[str], title: str) -> str:
    """Semantic identity of a pattern: model + category + the move chain.

    Two clusters that reach the same conclusion by the same route ARE the same
    strategy, however many distinct traces produced them. The old identity was
    ``sha256(model:category:trace_hashes)`` — addressed by the cluster's exact
    CONTENT, so every fresh batch of traces minted a "new" pattern and the bank
    filled with duplicate titles (measured 2026-08-26: 47% of the opus-5 slots).

    A chain shorter than ``_MIN_CHAIN_FOR_IDENTITY`` is NOT an identity: one
    move is a presence, not a sequence, so "the only move both clusters share
    is verify" says nothing about whether they are the same strategy. Those
    fall back to the title, as an empty chain does. Measured on the live bank:
    keying on a one-move chain would have folded 26 groups of DIFFERENT titles
    into one — "verification: verify, restate-goal, backtrack" and
    "verification: verify, restate-goal, gather-evidence" are not the same
    strategy. Requiring two moves drops that to 4 groups, all of which really
    do share a measured chain.
    """
    spine = ""
    if len([m for m in chain if m.strip()]) >= _MIN_CHAIN_FOR_IDENTITY:
        spine = " -> ".join(m.strip().lower() for m in chain if m.strip())
    if not spine:
        spine = "title:" + " ".join(title.strip().lower().split())
    return f"{model.strip().lower()}|{category.strip().lower()}|{spine}"


def chain_from_strategy(strategy: str) -> list[str]:
    """Recover the move chain from a stored ``_reasoning_strategy`` line.

    Needed for fibers written before the chain was stored structurally. Only a
    "Moves: a -> b" line carries a chain; "Moves (unordered): ..." and
    "Moves: (none detected)" mean there was none, and say so.
    """
    if not strategy:
        return []
    first = strategy.splitlines()[0]
    prefix = "Moves: "
    if not first.startswith(prefix):
        return []
    body = first[len(prefix) :].strip()
    if not body or body == "(none detected)":
        return []
    return [part.strip() for part in body.split("->") if part.strip()]


def _merge_pattern_into(fiber: Fiber, pattern: dict[str, Any], sig: str) -> Fiber | None:
    """Fold *pattern* into an existing twin fiber. None when it is already in.

    Returning None on a signature we already carry is what keeps distillation
    idempotent: replaying a batch must change nothing.
    """
    md = dict(fiber.metadata)
    sigs = [str(s) for s in (md.get("_reasoning_signatures") or [])]
    if not sigs and md.get("_reasoning_signature"):
        sigs = [str(md["_reasoning_signature"])]
    if sig in sigs:
        return None
    sigs.append(sig)

    old_freq = int(md.get("_reasoning_frequency", 0) or 0)
    new_freq = int(pattern["frequency"])
    old_conf = float(md.get("_reasoning_confidence", 0.0) or 0.0)
    new_conf = float(pattern["confidence"])
    total = old_freq + new_freq
    # Confidence is a SHARE of a category, so the two estimates have different
    # denominators and cannot simply be added or maxed: max() lets one small,
    # fully-clustered batch pin the pattern at 1.0 forever, and injection ranks
    # by confidence x frequency. Weight each estimate by the traces behind it.
    conf = (
        ((old_conf * old_freq) + (new_conf * new_freq)) / total
        if total
        else max(old_conf, new_conf)
    )

    md["_reasoning_signatures"] = sigs
    md["_reasoning_frequency"] = total
    md["_reasoning_confidence"] = round(min(1.0, conf), 4)
    if new_conf > old_conf:
        # Describe the pattern with its better-supported cluster. Title and
        # summary stay: they are this fiber's identity and the content of its
        # anchor CONCEPT neuron, which merging must not orphan.
        md["_reasoning_description"] = pattern["description"]
        md["_reasoning_strategy"] = pattern["strategy"]
    return replace(fiber, metadata=md)


async def _find_or_create_concept(
    storage: NeuralStorage, content: str, metadata: dict[str, Any] | None = None
) -> str:
    existing = await storage.find_neurons(content_exact=content, limit=1)
    if existing:
        return existing[0].id
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=content, metadata=metadata or {})
    await storage.add_neuron(neuron)
    return neuron.id


async def _materialize_pattern(
    storage: NeuralStorage,
    pattern: dict[str, Any],
    existing_sigs: set[str],
    existing_by_key: dict[str, Fiber],
) -> str:
    """Create — or MERGE INTO — the pattern fiber for *pattern*.

    Returns ``"created"``, ``"merged"`` or ``"noop"``. Idempotent twice over:
    by ``_reasoning_signature`` (this exact cluster was already materialized)
    and by ``merge_key`` (an equivalent strategy already has a fiber, so this
    cluster is folded into it instead of minting a near-duplicate).
    """
    sig = pattern["signature"]
    if sig in existing_sigs:
        return "noop"

    key = merge_key(
        pattern["model"], pattern["category"], pattern.get("chain") or (), pattern["title"]
    )
    twin = existing_by_key.get(key)
    if twin is not None:
        merged = _merge_pattern_into(twin, pattern, sig)
        if merged is None:
            return "noop"
        await storage.update_fiber(merged)
        existing_by_key[key] = merged
        return "merged"

    category_nid = await _find_or_create_concept(
        storage, f"reasoning_category:{pattern['category']}", {"_reasoning_category_concept": True}
    )
    pattern_nid = await _find_or_create_concept(
        storage, pattern["title"], {"_reasoning_pattern_concept": True}
    )

    existing_syn = await storage.get_synapses(
        source_id=pattern_nid, target_id=category_nid, type=SynapseType.EFFECTIVE_FOR
    )
    if existing_syn:
        synapse_id = existing_syn[0].id
    else:
        synapse = Synapse.create(
            source_id=pattern_nid,
            target_id=category_nid,
            type=SynapseType.EFFECTIVE_FOR,
            weight=min(1.0, float(pattern["confidence"])),
            direction=Direction.UNIDIRECTIONAL,
        )
        await storage.add_synapse(synapse)
        synapse_id = synapse.id

    fiber = Fiber.create(
        neuron_ids={pattern_nid, category_nid},
        synapse_ids={synapse_id},
        anchor_neuron_id=pattern_nid,
        pathway=[pattern_nid, category_nid],
        summary=pattern["title"],
        tags=set(),
        metadata={
            "_reasoning_pattern": True,
            "_source_model": pattern["model"],
            "_reasoning_category": pattern["category"],
            "_reasoning_title": pattern["title"],
            "_reasoning_description": pattern["description"],
            "_reasoning_strategy": pattern["strategy"],
            "_reasoning_frequency": pattern["frequency"],
            "_reasoning_confidence": pattern["confidence"],
            "_reasoning_signature": sig,
            # Every cluster folded into this fiber, so a replayed batch is a
            # no-op and the retro-merge can account for what it absorbed.
            "_reasoning_signatures": [sig],
            "_reasoning_chain": list(pattern.get("chain") or ()),
            "_reasoning_merge_key": key,
        },
    )
    # Patterns are activated only by injection (which may be OFF); unpinned they
    # are dead weight to decay/prune and vanish between sessions — pin them like
    # trained KB (doc_trainer) so lifecycle skips their neurons and fibers.
    fiber = replace(fiber, pinned=True)
    await storage.add_fiber(fiber)
    existing_by_key[key] = fiber
    return "created"


_LOCAL_SAFE_PROVIDERS: tuple[str, ...] = ("openai", "openrouter", "bge_m3")
_OLLAMA_DEFAULT_BASE = "http://localhost:11434"


def _warn_remote_endpoint(endpoint: str) -> None:
    """Say the embedder was refused, rather than degrading silently.

    A silent fallback to keyword classification is indistinguishable from
    "embeddings are off" from the outside, which is how a misconfigured
    endpoint went unnoticed long enough to freeze category coverage.
    """
    logger.warning(
        "reasoning distiller: embedding endpoint %r is not loopback — "
        "reasoning traces never leave this machine, so classification "
        "falls back to keywords. Point SURREAL_MEMORY_EMBEDDING_ENDPOINT "
        "or [embedding] endpoint at a local server to enable it.",
        endpoint or "<unset>",
    )


def _endpoint_is_loopback(endpoint: str) -> bool:
    """True when *endpoint* is a URL whose host is genuinely loopback.

    ``_is_loopback`` takes a HOST, not a URL — parsing is the caller's job (see
    ``resolve_llm_endpoint``), so handing it a full URL always answers False.
    The host test itself lives there and is shared, so both the distill-LLM and
    the embedder reach a remote endpoint under exactly one rule.
    """
    from urllib.parse import urlsplit

    if not endpoint.strip():
        return False
    try:
        return _is_loopback(urlsplit(endpoint.strip()).hostname)
    except ValueError:
        return False


def _get_embedder(config: UnifiedConfig | None = None) -> EmbeddingProvider | None:
    """Best-effort LOCAL embedding provider; None if unavailable (fail-soft).

    Only a local Ollama or a loopback OpenAI-compatible endpoint (llamastash
    bge-m3) is used, so distillation stays local + fast and never blocks on a
    remote/heavy provider. Any failure -> None and the caller falls back to
    keyword classification + move-set clustering.

    The CONFIGURED provider wins when embeddings are enabled. Deciding by
    environment probe alone meant an unrelated GEMINI_API_KEY export shadowed a
    correctly configured loopback endpoint: the probe answered "gemini", which
    this function cannot build, so it returned None and every trace was
    classified by keyword while a working bge-m3 sat idle. Delegating to the
    canonical factory also picks up the configured model name and the shared
    provider cache, neither of which the hand-rolled construction had.
    """
    if config is not None and config.embedding.enabled:
        provider = (config.embedding.provider or "").strip().lower()
        endpoint = config.embedding.resolved_endpoint()
        if provider == "auto":
            provider = ""  # fall through to the probe below
        elif provider == "ollama":
            # Ollama's own base URL, NOT the embedding endpoint: it is the value
            # this provider actually connects to, so it is the one that has to
            # clear the gate. Checking anything else would validate a string
            # that never reaches the socket.
            ollama_base = endpoint or os.environ.get("OLLAMA_BASE_URL", _OLLAMA_DEFAULT_BASE)
            if not _endpoint_is_loopback(ollama_base):
                _warn_remote_endpoint(ollama_base)
                return None
            try:
                from surreal_memory.engine.embedding.ollama_embedding import OllamaEmbedding

                return OllamaEmbedding(model=config.embedding.model, base_url=ollama_base)
            except Exception:
                logger.debug("reasoning distiller: ollama embedder could not be built")
                return None
        elif provider in _LOCAL_SAFE_PROVIDERS and _endpoint_is_loopback(endpoint):
            try:
                from surreal_memory.engine.embedding.openai_embedding import OpenAIEmbedding

                # base_url is passed EXPLICITLY so the endpoint that cleared the
                # gate is the endpoint the client connects to. Delegating to the
                # provider factory instead re-resolved it independently: an
                # openrouter provider carries a hardcoded remote default, and an
                # openai one reads only the env var, so a loopback endpoint set
                # in config.toml passed the check while traces went to the cloud.
                return OpenAIEmbedding(model=config.embedding.model, base_url=endpoint)
            except Exception:
                logger.debug(
                    "reasoning distiller: configured provider %r could not be built", provider
                )
                return None
        elif provider in _LOCAL_SAFE_PROVIDERS:
            _warn_remote_endpoint(endpoint)
            return None

    try:
        from surreal_memory.engine.semantic_discovery import _auto_detect_provider

        provider_name, model_name = _auto_detect_provider()
    except Exception:
        logger.debug("reasoning distiller: no embedding provider detected", exc_info=True)
        return None

    try:
        endpoint = os.environ.get("SURREAL_MEMORY_EMBEDDING_ENDPOINT", "")
        if provider_name == "ollama":
            # Same rule as the configured path: gate the URL this provider will
            # really open, which is OLLAMA_BASE_URL, not the embedding endpoint.
            ollama_base = os.environ.get("OLLAMA_BASE_URL", _OLLAMA_DEFAULT_BASE)
            if not _endpoint_is_loopback(ollama_base):
                _warn_remote_endpoint(ollama_base)
                return None
            from surreal_memory.engine.embedding.ollama_embedding import OllamaEmbedding

            return OllamaEmbedding(model=model_name, base_url=ollama_base)
        if provider_name in ("openai", "openrouter") and _endpoint_is_loopback(endpoint):
            from surreal_memory.engine.embedding.openai_embedding import OpenAIEmbedding

            return OpenAIEmbedding(model=model_name, base_url=endpoint)
    except Exception:
        logger.debug("reasoning distiller: embedding provider construction failed", exc_info=True)
    return None


async def _seed_centroids(
    embedder: EmbeddingProvider, categories: tuple[str, ...]
) -> dict[str, list[float]] | None:
    try:
        descriptions = [_CATEGORY_SEEDS.get(c, c) for c in categories]
        vectors = await embedder.embed_batch(descriptions)
        return {c: list(v) for c, v in zip(categories, vectors, strict=False)}
    except Exception:
        logger.debug("reasoning distiller: seed embedding failed", exc_info=True)
        return None


async def _embed_texts(
    embedder: EmbeddingProvider | None, texts: list[str]
) -> list[list[float]] | None:
    if embedder is None or not texts:
        return None
    try:
        return [list(v) for v in await embedder.embed_batch(texts)]
    except Exception:
        logger.debug("reasoning distiller: trace embedding failed", exc_info=True)
        return None


@dataclass
class DistillResult:
    """Outcome of a distillation pass."""

    patterns_learned: int = 0
    traces_processed: int = 0
    models_seen: int = 0
    # Clusters folded into an existing pattern instead of minting a duplicate.
    # Counted separately from patterns_learned: a merge is work done, and a run
    # that only merges must not be reportable as a run that did nothing.
    patterns_merged: int = 0


async def _process_model_batch(
    storage: NeuralStorage,
    brain_id: str,
    rt: ReasoningTrainingConfig,
    model: str,
    traces: list[dict[str, Any]],
    embedder: EmbeddingProvider | None,
    seeds: dict[str, list[float]] | None,
    existing_sigs: set[str],
    existing_by_key: dict[str, Fiber],
    budget: int,
    namer: PatternNamer | None = None,
) -> tuple[int, int, list[Any]]:
    """Distill one model's trace batch. Returns (created, merged, consumed_ids).

    ``consumed_ids`` are the traces safe to mark processed: ``other`` traces,
    under-support categories, and every category fully clustered before this
    model's remaining per-target ``budget`` ran out. A category left unreached —
    or cut off mid-cluster — by the budget is NOT consumed, so the next run
    revisits it (already-materialized patterns are skipped by signature).
    """
    clf_texts = [
        f"{t.get('task_context', '')} {str(t.get('content', ''))[:_CLASSIFY_CHARS]}".strip()
        for t in traces
    ]
    moves_list = [segment_moves(str(t.get("content", ""))) for t in traces]
    move_idf = _move_idf(moves_list)
    vectors = await _embed_texts(embedder, clf_texts)
    categories = [
        _classify_by_vector(vectors[i], seeds)
        if (vectors is not None and seeds)
        else _classify_by_keywords(clf_texts[i], rt.categories)
        for i in range(len(traces))
    ]
    await storage.set_trace_categories(
        brain_id, {t["id"]: categories[i] for i, t in enumerate(traces)}
    )

    consumed: list[Any] = [traces[i]["id"] for i, c in enumerate(categories) if c == _OTHER]
    by_category: dict[str, list[int]] = {}
    for i, cat in enumerate(categories):
        if cat != _OTHER:
            by_category.setdefault(cat, []).append(i)

    created = 0
    merged = 0
    for category, idxs in by_category.items():
        if created >= budget:
            break  # budget reached before this category → leave its traces unprocessed
        if len(idxs) < rt.min_cluster_support:
            consumed.extend(traces[i]["id"] for i in idxs)  # too few to cluster; done
            continue
        sub_vectors = [vectors[i] for i in idxs] if vectors is not None else None
        sub_moves = [moves_list[i] for i in idxs]
        sub_traces = [traces[i] for i in idxs]
        capped_mid_category = False
        for local_cluster in _cluster(sub_vectors, sub_moves, rt.cluster_cosine):
            if len(local_cluster) < rt.min_cluster_support:
                continue
            if created >= budget:
                capped_mid_category = True
                break
            cluster_traces = [sub_traces[k] for k in local_cluster]
            cluster_vecs = [sub_vectors[k] for k in local_cluster] if sub_vectors else None
            cluster_moves = [sub_moves[k] for k in local_cluster]
            pattern = _build_pattern(
                cluster_traces,
                cluster_vecs,
                cluster_moves,
                model,
                category,
                len(idxs),
                move_idf=move_idf,
            )
            if namer is not None:
                # Prose only: the signature is already fixed by the cluster's
                # trace hashes, so naming cannot fork a pattern into a duplicate.
                pattern = await namer.rename(pattern, cluster_traces)
            outcome = await _materialize_pattern(storage, pattern, existing_sigs, existing_by_key)
            if outcome != "noop":
                existing_sigs.add(pattern["signature"])
            if outcome == "created":
                created += 1
            elif outcome == "merged":
                # A merge fills no new slot, so it must not spend budget —
                # that is the point: folding duplicates frees room for genuinely
                # new strategies under an unchanged pattern_targets.
                merged += 1
        if capped_mid_category:
            break  # do NOT consume this category's traces — revisit next run
        consumed.extend(traces[i]["id"] for i in idxs)
    return created, merged, consumed


async def distill_reasoning_patterns(
    storage: NeuralStorage,
    brain_id: str,
    config: UnifiedConfig,
    *,
    embedder: EmbeddingProvider | None = None,
    drain: bool = False,
    progress: ProgressCallback | None = None,
) -> DistillResult:
    """Distill unprocessed reasoning traces into ReasoningBank pattern fibers.

    ``storage`` must already be on ``brain_id`` (graph writes use the current
    brain). Distillation is governed by per-model targets
    (``reasoning_training.pattern_targets``): for each detected source model the
    budget is ``max(0, target - existing_patterns_for_that_model)``. A model with
    budget 0 (its target is unset/0, or already met) is SKIPPED entirely — its
    traces stay unprocessed until a target is raised, so a preliminary Mine with
    no targets set only DETECTS models without distilling anything.

    ``drain=True`` (a manual ``POST /mine``) keeps fetching batches for a model
    until its budget is spent or its backlog is exhausted; ``drain=False``
    (background consolidation) processes at most one batch per model per run.
    Consumed traces are marked processed per batch so the next fetch returns
    fresh work. ``progress`` receives a distilling snapshot as each model
    advances.
    """
    rt = config.reasoning_training
    embedder = embedder or _get_embedder(config)
    seeds = await _seed_centroids(embedder, rt.categories) if embedder is not None else None
    # None unless distill_use_llm is on AND a local endpoint and model are set.
    # acquire() explicitly loads the chat model when distill_llm_load_cmd is
    # configured (a no-op otherwise, falling back to the first rename pulling
    # it in implicitly as before); released in the finally below either way,
    # so it is resident for this run only.
    namer = build_namer(rt)
    if namer is not None:
        await namer.acquire()

    existing = await storage.find_fibers(
        metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT
    )
    existing_sigs: set[str] = set()
    existing_by_key: dict[str, Fiber] = {}
    for f in existing:
        md = f.metadata
        if md.get("_reasoning_signature"):
            existing_sigs.add(str(md["_reasoning_signature"]))
        existing_sigs.update(str(s) for s in (md.get("_reasoning_signatures") or ()))
        # Fibers written before the chain was stored structurally carry it only
        # in the rendered strategy line — recover it there rather than treating
        # the whole legacy bank as unmergeable.
        key = str(md.get("_reasoning_merge_key") or "") or merge_key(
            str(md.get("_source_model") or ""),
            str(md.get("_reasoning_category") or ""),
            chain_from_strategy(str(md.get("_reasoning_strategy") or "")),
            str(md.get("_reasoning_title") or ""),
        )
        existing_by_key.setdefault(key, f)

    existing_by_model: dict[str, int] = {}
    for f in existing:
        source_model = f.metadata.get("_source_model")
        if source_model:
            existing_by_model[str(source_model)] = existing_by_model.get(str(source_model), 0) + 1

    patterns_created = 0
    patterns_merged = 0
    processed_ids: list[Any] = []
    models = await storage.get_reasoning_trace_models(brain_id)
    if rt.mining_models:
        # Honor the configured source-model globs so distillation is restricted to
        # the same models as ingestion (and to POST /mine's models= override). An
        # empty mining_models means "all models" (unchanged default behavior).
        models = [m for m in models if any(fnmatch(m, pat) for pat in rt.mining_models)]
    models_total = len(models)

    def _emit(current_model: str | None, models_done: int) -> None:
        if progress is not None:
            progress(
                MiningProgress(
                    phase=PHASE_DISTILLING,
                    traces_processed=len(processed_ids),
                    patterns_learned=patterns_created,
                    current_model=current_model,
                    models_done=models_done,
                    models_total=models_total,
                )
            )

    try:
        for idx, model in enumerate(models):
            budget = max(0, rt.pattern_targets.get(model, 0) - existing_by_model.get(model, 0))
            if budget <= 0:
                # Target unset/0 or already met → leave this model's traces unprocessed.
                _emit(model, idx + 1)
                continue
            while budget > 0:
                traces = await storage.get_unprocessed_reasoning_traces(
                    brain_id, limit=_BATCH_PER_MODEL, model=model
                )
                traces = traces[:_BATCH_PER_MODEL]
                if not traces:
                    break  # backlog for this model exhausted
                created, merged, consumed = await _process_model_batch(
                    storage,
                    brain_id,
                    rt,
                    model,
                    traces,
                    embedder,
                    seeds,
                    existing_sigs,
                    existing_by_key,
                    budget,
                    namer,
                )
                patterns_created += created
                patterns_merged += merged
                budget -= created
                if consumed:
                    processed_ids.extend(consumed)
                    # Mark consumed processed NOW so the next fetch returns fresh
                    # traces — otherwise a drain loop re-fetches the same batch forever.
                    await storage.mark_reasoning_traces_processed(brain_id, consumed)
                _emit(model, idx)
                # Termination guard: a batch that consumes nothing makes no forward
                # progress (budget hit 0 mid-category), so stop draining this model.
                if not consumed:
                    break
                if not drain:
                    break  # background consolidation: one batch per model per run
            _emit(model, idx + 1)
    finally:
        # Unconditional: an aborted or failed run must not leave the chat model
        # parked in VRAM either.
        if namer is not None:
            await namer.release()

    if processed_ids:
        await storage.prune_reasoning_traces(brain_id, rt.retention_days)
        await storage.cap_reasoning_traces(brain_id, rt.max_traces_total)

    return DistillResult(
        patterns_learned=patterns_created,
        traces_processed=len(processed_ids),
        models_seen=models_total,
        patterns_merged=patterns_merged,
    )


async def reasoning_coverage(
    storage: NeuralStorage,
    model: str,
    config: UnifiedConfig,
) -> dict[str, Any]:
    """Per-category coverage for *model*.

    A category is covered iff it has >= ``min_patterns_per_category`` pattern
    fibers with ``_source_model == model`` and confidence >= ``min_confidence``.
    ``coverage_percent`` = covered / len(categories) * 100 (``other`` excluded —
    it is never in ``categories``). ``storage`` must be on the target brain.
    """
    rt = config.reasoning_training
    fibers = await storage.find_fibers(
        metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT
    )
    counts: dict[str, int] = dict.fromkeys(rt.categories, 0)
    for f in fibers:
        md = f.metadata
        if md.get("_source_model") != model:
            continue
        if float(md.get("_reasoning_confidence", 0.0)) < rt.min_confidence:
            continue
        cat = md.get("_reasoning_category")
        if cat in counts:
            counts[cat] += 1
    covered = {c: counts[c] >= rt.min_patterns_per_category for c in rt.categories}
    n_covered = sum(1 for v in covered.values() if v)
    percent = (n_covered / len(rt.categories) * 100.0) if rt.categories else 0.0
    return {"by_category": counts, "covered": covered, "coverage_percent": round(percent, 1)}
