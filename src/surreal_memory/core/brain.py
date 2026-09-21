"""Brain container - the top-level memory structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from surreal_memory.utils.timeutils import utcnow


@dataclass(frozen=True)
class BrainConfig:
    """
    Configuration for brain behavior.

    Attributes:
        decay_rate: Rate at which neuron activation decays (per day)
        reinforcement_delta: Amount to increase synapse weight on access
        activation_threshold: Minimum activation level to consider active
        max_spread_hops: Maximum hops in spreading activation
        max_context_tokens: Maximum tokens to include in context injection
        default_synapse_weight: Default weight for new synapses
    """

    decay_rate: float = 0.1
    reinforcement_delta: float = 0.05
    # How many of a recall's top-activated neurons get reinforced and fed into
    # maturation rehearsal. Was hardcoded to 10 in two independent places
    # (retrieval's top-K selection and the rehearsal fan-out), so a recall on
    # any brain rehearsed at most 10 fibers regardless of how many it actually
    # activated -- capping the EPISODIC->SEMANTIC spacing gate's only
    # rehearsal source at a size that did not scale with the brain. 15 (up
    # from the old 10) rather than a larger jump: measured against a live
    # SurrealDB, each additional neuron here costs ~24ms (find_fibers_batch's
    # per-neuron sequential lookup has no index on fiber.neuron_ids), so this
    # is deliberately conservative -- raise it in config.toml if your setup
    # can absorb the added recall latency.
    reinforcement_neuron_limit: int = 15
    activation_threshold: float = 0.2
    max_spread_hops: int = 4
    max_context_tokens: int = 1500
    default_synapse_weight: float = 0.5
    hebbian_delta: float = 0.03
    hebbian_threshold: float = 0.5
    hebbian_initial_weight: float = 0.2
    consolidation_prune_threshold: float = 0.05
    prune_min_inactive_days: float = 7.0
    merge_overlap_threshold: float = 0.5
    sigmoid_steepness: float = 6.0
    default_firing_threshold: float = 0.3
    default_refractory_ms: float = 500.0
    lateral_inhibition_k: int = 10
    lateral_inhibition_factor: float = 0.3
    learning_rate: float = 0.05
    weight_normalization_budget: float = 5.0
    novelty_boost_max: float = 3.0
    novelty_decay_rate: float = 0.06
    co_activation_threshold: int = 3
    co_activation_window_days: int = 7
    max_inferences_per_run: int = 50
    emotional_decay_factor: float = 0.5
    emotional_weight_scale: float = 0.8
    sequential_window_seconds: float = 30.0
    dream_neuron_count: int = 5
    dream_decay_multiplier: float = 10.0
    habit_min_frequency: int = 3
    habit_suggestion_min_weight: float = 0.8
    habit_suggestion_min_count: int = 5
    embedding_enabled: bool = False
    embedding_provider: str = "sentence_transformer"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_similarity_threshold: float = 0.7
    embedding_activation_boost: float = 0.15
    freshness_weight: float = 0.0
    semantic_discovery_similarity_threshold: float = 0.7
    semantic_discovery_max_pairs: int = 2000
    # Adaptive recall (Bayesian depth priors)
    adaptive_depth_enabled: bool = True
    adaptive_depth_epsilon: float = 0.05
    # Memory compression
    compression_enabled: bool = True
    compression_tier_thresholds: tuple[int, ...] = (7, 30, 90, 180)
    # Retrieval: Reciprocal Rank Fusion
    rrf_k: int = 60
    # Retrieval: Graph-based query expansion
    graph_expansion_enabled: bool = True
    graph_expansion_max: int = 10
    graph_expansion_min_weight: float = 0.3
    # Retrieval: fiber-level vector anchors (N2 fix, smem-recall-leksyka-fibry-reranker,
    # 2026-09-12) — measured +5/49 golden hits, the only retriever that reaches a fiber
    # without going through one of its neurons as an anchor. Off by default: it needs
    # `scripts/backfill_fiber_vectors.py` to have populated `fiber.fiber_vec` first, and
    # enabling it on a brain without that backfill contributes nothing (find_fibers_by_embedding
    # simply returns no usable rows), so there is no silent behavior change on existing brains.
    fiber_vector_enabled: bool = False
    fiber_vector_top_n: int = 10
    # Retrieval: similarity floor for fiber-vector anchors specifically (smem-recall-tor-
    # fibrowy, U2-REVISIT/B5). Until this key existed, the fiber track reused
    # `embedding_similarity_threshold` (below) — but that value also gates the neuron-vector
    # track's `_rank_knn_rows`, and the two tracks need different values: measured on the
    # production golden set, the two fiber anchors that recover a lost golden pair have
    # `_sim` 0.4879 and 0.5027 (ABBA measurement, `qa/anatomia-kotwic-*.json` in the program
    # ledger), so 0.52 drops both while 0.45 keeps both with headroom. Lowering the SHARED
    # `embedding_similarity_threshold` to 0.45 instead would "fix" the fiber track but also
    # loosens the neuron track's KNN filter, which measurably let a negative-control query
    # accumulate anchors it should not have had — hence a track-specific key rather than a
    # shared one. 0.45 is not "disable the filter": it is the measured floor with headroom
    # below the lower of the two recovered similarities (0.4879), not 0.
    fiber_vector_similarity_threshold: float = 0.45
    # Retrieval: minimum content length for a keyword anchor candidate (N1 fix,
    # smem-recall-leksyka-fibry-reranker, 2026-09-12). `find_neurons_ranked` orders keyword
    # anchors by BM25, which is length-biased (b=0.75), so very short neurons score high on a
    # single term match without carrying the context that makes an anchor useful. 25 characters
    # is the measured value: it tied with the un-gated variant on the 49-pair golden and lost
    # nothing, so it is cheap insurance rather than a tuned threshold. Configurable because it
    # is an empirical number on one brain's content — another brain should be able to move it
    # without a release. 0 disables the gate.
    keyword_anchor_min_content_len: int = 25
    # Retrieval: Activation strategy
    activation_strategy: str = "classic"  # "ppr" | "classic" | "reflex" | "hybrid" | "auto"
    ppr_damping: float = 0.15
    ppr_iterations: int = 20
    ppr_epsilon: float = 1e-6
    # Cascading retrieval: fiber summary tier + sufficiency gate
    fiber_summary_tier_enabled: bool = True
    sufficiency_threshold: float = 0.7
    # Lazy entity promotion: entities need N mentions to become neurons
    lazy_entity_enabled: bool = True
    lazy_entity_promotion_threshold: int = 2
    lazy_entity_prune_days: int = 90
    # Lazy concept promotion: same idea for keyword/concept neurons, which until now
    # became permanent on their FIRST appearance while entities had to earn it. Because
    # the keyword extractor emits mostly adjacent-word bi-grams, that asymmetry is what
    # fills a brain with debris like "normy przedmiarowej" or "architektura silnika".
    # Measured on a production brain (1199 real memories, replayed): 82 % of concept
    # creations were for a keyword that never recurred; at threshold 3 it is 92 %.
    # Recurrence is read from the keyword_document_frequency table that the encoder
    # already maintains, so nothing new has to be tracked. The memory itself is stored
    # either way; only the keyword's own index entry waits for a second mention.
    #
    # NOT full parity with the entity path, and the difference is lossy: entities get
    # _retroactive_entity_link() so earlier memories are wired to an entity once it is
    # promoted, while a promoted concept is linked only from the memory that promoted it.
    # The FIRST memory mentioning a keyword permanently loses that one activation hop.
    # Accepted deliberately -- it costs an index edge, not content, and the anchor keeps
    # both its text and its embedding -- but a retroactive concept link is the obvious
    # next step if recall quality on first-mention memories ever regresses.
    lazy_concept_enabled: bool = True
    lazy_concept_promotion_threshold: int = 2
    # Recall quality: recency sigmoid halflife (hours)
    recency_halflife_hours: float = 168.0  # 7 days (was hardcoded 72h)
    # Recall quality: tag-aware scoring boost
    tag_match_boost: float = 0.15
    # Prune: dead neuron minimum age (days) before auto-prune
    prune_dead_neuron_days: float = 14.0
    # Diminishing returns gate: stop spreading when new hops add little signal
    diminishing_returns_enabled: bool = True
    diminishing_returns_threshold: float = 0.15
    diminishing_returns_min_neurons: int = 2
    diminishing_returns_grace_hops: int = 1
    # Fidelity layers
    decay_floor: float = 0.05
    fidelity_enabled: bool = True
    fidelity_full_threshold: float = 0.6
    fidelity_summary_threshold: float = 0.3
    fidelity_essence_threshold: float = 0.1
    essence_generator: str = "extractive"  # "extractive" or "llm"
    # Fuzzy search (typo tolerance)
    fuzzy_search_enabled: bool = False
    fuzzy_search_max_distance: int = 2
    fuzzy_search_max_candidates: int = 50
    # IDF-weighted anchor selection
    idf_anchor_enabled: bool = False
    idf_anchor_min_limit: int = 1
    idf_anchor_max_limit: int = 5
    # Query expansion (synonym, abbreviation)
    query_expansion_synonyms: bool = True
    query_expansion_abbreviations: bool = True
    query_expansion_max_per_term: int = 5
    # Cross-encoder reranking (optional post-SA refinement)
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_overfetch_multiplier: int = 3
    reranker_blend_weight: float = 0.7  # Reranker weight (SA gets 1 - this)
    reranker_min_score: float = 0.15
    reranker_max_candidates: int = 30  # Safety cap on overfetch
    reranker_endpoint: str = (
        ""  # OpenAI-compatible /rerank base URL (llamastash); empty = in-process CrossEncoder
    )
    # Temporal binding (session-level auto-linking)
    temporal_binding_enabled: bool = True
    temporal_binding_window_seconds: float = 300.0  # 5-minute window
    # Arousal detection (emotional intensity for compression resistance)
    arousal_enabled: bool = True
    # Prediction error encoding (surprise signal boosts priority)
    prediction_error_enabled: bool = True
    # Retrieval reconsolidation (recalled memories absorb context)
    reconsolidation_enabled: bool = True
    reconsolidation_drift_threshold: float = 0.6
    # Context-dependent retrieval (project-scoped scoring)
    context_retrieval_enabled: bool = True
    # Hippocampal replay consolidation (LTP/LTD)
    replay_enabled: bool = True
    replay_ltp_factor: float = 1.1
    replay_ltd_factor: float = 0.98
    # Working memory chunking (group retrieval output)
    chunking_enabled: bool = True
    max_chunks: int = 5
    # Schema assimilation (bottom-up knowledge organization)
    schema_assimilation_enabled: bool = False
    schema_min_cluster_size: int = 10
    # Interference forgetting (memory competition detection)
    interference_detection_enabled: bool = False
    fan_effect_threshold: int = 15
    # Trust/recency calibration (U2 — opt-in; neutral defaults preserve ranking)
    trust_weight: float = 0.0  # 0.0 = trust ignored in final scoring (default no-op)
    recency_weight: float = 1.0  # 1.0 = existing recency decay unchanged (default no-op)
    trust_default: float = 0.7  # fallback trust when no per-memory/source signal resolves
    # Retrieval recency anchor: fall back to `created_at` when a fiber was never
    # recalled. Without it `last_conducted is None` scores a flat 0.5, so a memory
    # written minutes ago starts *below* one recalled yesterday (≈0.85 at the 168 h
    # half-life) — the ranking rewards rehearsal and is blind to age. Fibers with
    # neither timestamp keep the historical 0.5.
    recency_from_created: bool = True
    # Retrieval priority weighting. `priority` was stored (typed_memory, and as
    # `auto_priority` in fiber metadata) but read by nothing in scoring, so marking a
    # memory as critical had no effect on recall. Multiplier is
    # `1 + weight * (p - 5) / 5` with p clamped to [0, 10]: neutral at the default
    # priority 5, ±weight at the extremes — a tie-breaker among near-equals, smaller
    # than the recency swing. `auto_priority` is machine-derived novelty, not human
    # importance, so it gets its own weight and stays inert unless asked for.
    priority_weight: float = 0.2
    auto_priority_weight: float = 0.0
    # How the semantic retriever picks anchor neurons.
    # "scan" is the historical path: read the first `find_neurons` page and
    # score it in Python. That page is ordered by id and capped, so on a brain
    # larger than the cap the semantic retriever only ever sees its oldest
    # slice. "knn" asks the backend's vector index for the actual nearest
    # neighbours. "auto" uses the index when the backend has one and falls
    # back to the scan otherwise, saying so in the retrieval metadata.
    embedding_anchor_mode: str = "auto"
    # Retrieval: refusal gate on the cross-encoder's raw top-1 score (M4,
    # smem-recall-brama-odmowy, U2 DIAGNOZA.md §7/§8). Recall's sufficiency
    # gates (`engine/sufficiency.py`) score the SHAPE of the activation
    # landscape, not whether it answers the query — measured on a 27-phrase
    # out-of-base set plus the 49-pair golden, every existing gate accepted
    # unconditionally (`default_pass`, 0/98 refusals possible today). The
    # reranker's raw cross-encoder score is the one signal in the pipeline
    # that reads the (query, content) pair itself: at the log-margin
    # threshold 0.002146 (5% of the golden range in log-space — the raw
    # score spans ~342x, so a linear margin degenerates below the whole
    # measured population) it refuses 17/27 out-of-base phrases while
    # refusing ZERO of the 98 golden queries (golden+pudła), AUC 0.9728
    # (highest of the ten signals measured). `None` = gate inactive — an
    # old brain, or a brain whose operator has not measured its own floor,
    # sees exactly today's behaviour (no new refusals). Wired in
    # `engine/retrieval.py` after 4.9 (post-rerank): skipped, never firing
    # a refusal, when the reranker itself degraded that query
    # (`metadata.reranker_floor_skipped` records why).
    reranker_refusal_floor: float | None = None
    # Retrieval: cheap pre-reranker refusal floor on activated neuron count
    # (weak_landscape_floor gate, `engine/sufficiency.py`, same DIAGNOZA.md
    # §7 measurement as above). `suff_neuron_count < 15` alone refuses
    # 19/27 out-of-base phrases at 0/98 golden refusals (AUC 0.9637) — the
    # single best individual signal measured, available at step 4.8 before
    # the reranker's ~1.2s cost. `0` = gate inactive (neuron_count is never
    # negative, so `neuron_count < 0` can never fire) — an old brain keeps
    # today's behaviour.
    sufficiency_min_neuron_count: int = 0
    # Retrieval: cheap pre-reranker refusal floor on the CLOSEST embedding
    # neighbour's similarity, BEFORE `embedding_similarity_threshold` is
    # applied (weak_landscape_floor gate, same measurement) — deliberately
    # not the similarity of the best-ranked ANCHOR. `embedding_similarity_
    # threshold` governs anchor SELECTION; this floor asks a different
    # question (how close the nearest neighbour is at all), and gating it
    # on the same threshold made it unable to ever fire (U3-REVISIT):
    # measured on DIAGNOZA.md's 54-row negative set, every one of the 36
    # rows this floor is meant to refuse has its closest neighbour below
    # `embedding_similarity_threshold` (0.52) — a threshold-filtered value
    # would be `None` ("inactive") for all 36. `anchor_sim_top1 <
    # 0.524835` alone refuses 18/27 out-of-base phrases at 0/98 golden
    # refusals (AUC 0.9403); combined with `sufficiency_min_neuron_count`
    # via OR it reaches 21/27 — KRYTERIUM OS3, the best ≤2-signal
    # combination measured (DIAGNOZA.md §3/§6). `None` = gate inactive (an
    # old brain, or a query where the embedding retriever had no row to
    # measure at all — never ran, or every KNN row was a tombstone; the
    # condition is skipped, not treated as a similarity of 0).
    sufficiency_min_anchor_sim: float | None = None
    # Retrieval: master mode for the refusal gates (weak_landscape_floor in
    # `engine/sufficiency.py`, the post-rerank M4 floor in
    # `engine/retrieval.py`), program `smem-recall-trzy-warstwy`. One of
    # "off" | "observe" | "enforce" — corrected 2026-09-21 (runner round 2)
    # after "off" was found to silently disable an OPERATOR-configured
    # enforcement knob (`reranker_refusal_floor` etc.) on any existing
    # brain that did not also set this new field:
    #   "off" (default) and "enforce" are DELIBERATE SYNONYMS — both read
    #     ONLY the enforcement knobs below (`sufficiency_min_neuron_count`,
    #     `sufficiency_min_anchor_sim`, `reranker_refusal_floor`) exactly as
    #     this gate worked before this field existed: a knob left at its
    #     own inactive default (0 / None) still never fires, but a knob an
    #     operator explicitly set keeps refusing with ZERO extra opt-in.
    #     Any value other than "observe" (a typo included) falls into this
    #     same safe bucket — never a silent disable.
    #   "observe" is the ONLY mode that changes control flow: the gates
    #     evaluate their OWN, separate, independent thresholds
    #     (`refusal_observe_min_neuron_count` / `refusal_observe_min_anchor_
    #     sim` / `refusal_observe_rerank_floor` below) and NEVER refuse —
    #     the client sees the same answer as "off"/"enforce" while
    #     `SufficiencyResult.would_refuse`/`would_refuse_gate`/`signals` and
    #     `RetrievalResult.metadata["odmowa_sygnaly"]` record what the
    #     decision WOULD have been. The enforcement knobs above are not
    #     even read in this mode — enabling observation can never
    #     accidentally start enforcing.
    refusal_mode: str = "off"
    # Program smem-recall-trzy-warstwy — thresholds used ONLY by
    # `refusal_mode="observe"` to compute "would refuse" signals; NEVER
    # read outside "observe", and NEVER cause an actual refusal (mandate:
    # "wyłącznie tryb obserwacji, wszystkie gałki egzekwowania OFF" — so
    # turning observation on must never require touching the enforcement
    # knobs above). Defaults are the W3 variant measured by program
    # smem-recall-brama-odmowy (`~/expertP/smem-recall-brama-odmowy/qa/
    # D1.md` §2a: "W3 (W2 + `reranker_refusal_floor=0.002146`)", where
    # "W2" is `sufficiency_min_neuron_count=15` LUB `sufficiency_min_
    # anchor_sim=0.524835`) — the same numbers as the enforcement knobs'
    # own DIAGNOZA.md measurement, just wired to a mode that can never
    # enforce.
    refusal_observe_min_neuron_count: int = 15
    refusal_observe_min_anchor_sim: float = 0.524835
    refusal_observe_rerank_floor: float = 0.002146

    def with_updates(self, **kwargs: Any) -> BrainConfig:
        """Create a new config with updated values."""
        return dc_replace(self, **kwargs)


@dataclass(frozen=True)
class Brain:
    """
    A Brain is the top-level container for a memory system.

    It holds configuration, ownership, and statistics for a
    collection of neurons, synapses, and fibers.

    Attributes:
        id: Unique identifier
        name: Human-readable name
        config: Brain configuration settings
        owner_id: Optional owner identifier
        is_public: Whether this brain can be read by anyone
        shared_with: List of user IDs with access
        neuron_count: Number of neurons (computed)
        synapse_count: Number of synapses (computed)
        fiber_count: Number of fibers (computed)
        metadata: Additional brain-specific data
        created_at: When this brain was created
        updated_at: When this brain was last modified
    """

    id: str
    name: str
    config: BrainConfig = field(default_factory=BrainConfig)
    owner_id: str | None = None
    is_public: bool = False
    shared_with: list[str] = field(default_factory=list)
    neuron_count: int = 0
    synapse_count: int = 0
    fiber_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    @classmethod
    def create(
        cls,
        name: str,
        config: BrainConfig | None = None,
        owner_id: str | None = None,
        is_public: bool = False,
        brain_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Brain:
        """
        Factory method to create a new Brain.

        Args:
            name: Human-readable name
            config: Optional configuration (uses defaults if None)
            owner_id: Optional owner identifier
            is_public: Whether publicly accessible
            brain_id: Optional explicit ID
            metadata: Optional metadata

        Returns:
            A new Brain instance
        """
        return cls(
            id=brain_id or str(uuid4()),
            name=name,
            config=config or BrainConfig(),
            owner_id=owner_id,
            is_public=is_public,
            metadata=metadata or {},
            created_at=utcnow(),
            updated_at=utcnow(),
        )

    def share_with(self, user_id: str) -> Brain:
        """
        Create a new Brain shared with an additional user.

        Args:
            user_id: User ID to share with

        Returns:
            New Brain with updated shared_with list
        """
        if user_id in self.shared_with:
            return self

        return Brain(
            id=self.id,
            name=self.name,
            config=self.config,
            owner_id=self.owner_id,
            is_public=self.is_public,
            shared_with=[*self.shared_with, user_id],
            neuron_count=self.neuron_count,
            synapse_count=self.synapse_count,
            fiber_count=self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def unshare_with(self, user_id: str) -> Brain:
        """
        Create a new Brain with a user removed from sharing.

        Args:
            user_id: User ID to remove

        Returns:
            New Brain with updated shared_with list
        """
        return Brain(
            id=self.id,
            name=self.name,
            config=self.config,
            owner_id=self.owner_id,
            is_public=self.is_public,
            shared_with=[uid for uid in self.shared_with if uid != user_id],
            neuron_count=self.neuron_count,
            synapse_count=self.synapse_count,
            fiber_count=self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def make_public(self) -> Brain:
        """Create a new Brain that is publicly accessible."""
        return Brain(
            id=self.id,
            name=self.name,
            config=self.config,
            owner_id=self.owner_id,
            is_public=True,
            shared_with=self.shared_with,
            neuron_count=self.neuron_count,
            synapse_count=self.synapse_count,
            fiber_count=self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def make_private(self) -> Brain:
        """Create a new Brain that is private."""
        return Brain(
            id=self.id,
            name=self.name,
            config=self.config,
            owner_id=self.owner_id,
            is_public=False,
            shared_with=self.shared_with,
            neuron_count=self.neuron_count,
            synapse_count=self.synapse_count,
            fiber_count=self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def with_config(self, config: BrainConfig) -> Brain:
        """Create a new Brain with updated configuration."""
        return Brain(
            id=self.id,
            name=self.name,
            config=config,
            owner_id=self.owner_id,
            is_public=self.is_public,
            shared_with=self.shared_with,
            neuron_count=self.neuron_count,
            synapse_count=self.synapse_count,
            fiber_count=self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def with_stats(
        self,
        neuron_count: int | None = None,
        synapse_count: int | None = None,
        fiber_count: int | None = None,
    ) -> Brain:
        """Create a new Brain with updated statistics."""
        return Brain(
            id=self.id,
            name=self.name,
            config=self.config,
            owner_id=self.owner_id,
            is_public=self.is_public,
            shared_with=self.shared_with,
            neuron_count=neuron_count if neuron_count is not None else self.neuron_count,
            synapse_count=synapse_count if synapse_count is not None else self.synapse_count,
            fiber_count=fiber_count if fiber_count is not None else self.fiber_count,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=utcnow(),
        )

    def can_access(self, user_id: str | None) -> bool:
        """
        Check if a user can access this brain.

        Args:
            user_id: User ID to check (None for anonymous)

        Returns:
            True if user has access
        """
        if self.is_public:
            return True
        if user_id is None:
            return False
        if self.owner_id == user_id:
            return True
        return user_id in self.shared_with

    def can_write(self, user_id: str | None) -> bool:
        """
        Check if a user can write to this brain.

        Args:
            user_id: User ID to check (None for anonymous)

        Returns:
            True if user has write access
        """
        if user_id is None:
            return False
        return self.owner_id == user_id


@dataclass(frozen=True)
class BrainSnapshot:
    """
    A serializable snapshot of a brain for export/import.

    Attributes:
        brain_id: ID of the original brain
        brain_name: Name of the brain
        exported_at: When this snapshot was created
        version: Schema version for compatibility
        neurons: List of serialized neurons
        synapses: List of serialized synapses
        fibers: List of serialized fibers
        config: Brain configuration
        metadata: Additional export metadata
    """

    brain_id: str
    brain_name: str
    exported_at: datetime
    version: str
    neurons: list[dict[str, Any]]
    synapses: list[dict[str, Any]]
    fibers: list[dict[str, Any]]
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
