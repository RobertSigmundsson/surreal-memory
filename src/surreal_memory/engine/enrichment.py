"""Enrichment engine — transitive closure and cross-cluster linking.

Creates new knowledge from existing knowledge:
- Transitive closure: If A→B→C (CAUSED_BY), infer A→C
- Cross-cluster linking: Connect clusters sharing entity neurons
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.clustering import UnionFind

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage


@dataclass(frozen=True)
class EnrichmentResult:
    """Result of enrichment operations.

    Attributes:
        transitive_synapses: New synapses from transitive closure
        cross_cluster_synapses: New synapses from cross-cluster linking
    """

    transitive_synapses: tuple[Synapse, ...] = field(default_factory=tuple)
    cross_cluster_synapses: tuple[Synapse, ...] = field(default_factory=tuple)

    @property
    def total_synapses(self) -> int:
        return len(self.transitive_synapses) + len(self.cross_cluster_synapses)


async def find_transitive_closures(
    storage: NeuralStorage,
    max_depth: int = 2,
    max_synapses: int = 50,
) -> list[Synapse]:
    """Find transitive closure opportunities in CAUSED_BY chains.

    For each A→B→C chain where no A→C exists, create a new
    CAUSED_BY synapse with weight = 0.5 * min(w_AB, w_BC).

    Args:
        storage: Storage backend
        max_depth: Maximum chain depth to traverse (default: 2 = A→B→C)
        max_synapses: Maximum new synapses to create

    Returns:
        List of new Synapse objects (not yet persisted)
    """
    causal_synapses = await storage.get_synapses(type=SynapseType.CAUSED_BY)
    if not causal_synapses:
        return []

    # Build adjacency: source_id -> [(target_id, weight)]
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for syn in causal_synapses:
        adjacency[syn.source_id].append((syn.target_id, syn.weight))

    # Build existing pairs for duplicate checking
    existing_pairs: set[tuple[str, str]] = set()
    for syn in causal_synapses:
        existing_pairs.add((syn.source_id, syn.target_id))

    new_synapses: list[Synapse] = []

    # Find A→B→C chains (depth=2)
    for a_id, a_targets in adjacency.items():
        if len(new_synapses) >= max_synapses:
            break
        for b_id, w_ab in a_targets:
            if len(new_synapses) >= max_synapses:
                break
            for c_id, w_bc in adjacency.get(b_id, []):
                if len(new_synapses) >= max_synapses:
                    break
                if a_id == c_id:
                    continue
                if (a_id, c_id) in existing_pairs:
                    continue

                weight = 0.5 * min(w_ab, w_bc)
                synapse = Synapse.create(
                    source_id=a_id,
                    target_id=c_id,
                    type=SynapseType.CAUSED_BY,
                    weight=weight,
                    metadata={"_enriched": True, "_chain": [a_id, b_id, c_id]},
                )
                new_synapses.append(synapse)
                existing_pairs.add((a_id, c_id))

    return new_synapses


async def find_cross_cluster_links(
    storage: NeuralStorage,
    tag_overlap_threshold: float = 0.4,
    max_synapses: int = 50,
) -> list[Synapse]:
    """Find cross-cluster linking opportunities.

    Clusters fibers by tag Jaccard similarity. For pairs of clusters
    sharing entity neurons, creates RELATED_TO between cluster anchors.

    Args:
        storage: Storage backend
        tag_overlap_threshold: Minimum Jaccard overlap for same cluster
        max_synapses: Maximum new synapses to create

    Returns:
        List of new Synapse objects (not yet persisted)
    """
    fibers = await storage.get_fibers(limit=10000)
    tagged_fibers = [f for f in fibers if f.tags]
    if len(tagged_fibers) < 2:
        return []

    # Cap fiber count for O(N^2) Jaccard comparison — 1000 fibers = 500K pairs max
    max_fibers_for_clustering = 1000
    if len(tagged_fibers) > max_fibers_for_clustering:
        # Keep highest-salience fibers
        tagged_fibers = sorted(tagged_fibers, key=lambda f: f.salience, reverse=True)[
            :max_fibers_for_clustering
        ]

    n = len(tagged_fibers)

    # Union-Find clustering
    uf = UnionFind(n)

    for i in range(n):
        tags_a = tagged_fibers[i].tags
        if not tags_a:
            continue
        # Yield to event loop every 20 outer iterations so timeout can fire
        if i % 20 == 0:
            await asyncio.sleep(0)
        for j in range(i + 1, n):
            tags_b = tagged_fibers[j].tags
            if not tags_b:
                continue
            intersection = len(tags_a & tags_b)
            if intersection == 0:
                continue
            union_size = len(tags_a) + len(tags_b) - intersection
            if union_size > 0 and intersection / union_size >= tag_overlap_threshold:
                uf.union(i, j)

    # Group fibers by cluster root
    cluster_list = [indices for indices in uf.groups().values() if len(indices) >= 1]
    if len(cluster_list) < 2:
        return []

    # Referential liveness guard (audit 2026-07-27): fibers outlive their
    # neurons — cascade_delete_synapses cleans synapses on neuron DELETE, but
    # nothing re-anchors fibers, so anchor_neuron_id / neuron_ids may point at
    # deleted records. Anchoring a RELATED_TO on such an id creates a synapse
    # that is orphaned at birth (26 measured in prod, all from this function),
    # and because the existing-pair dedup below only sees live synapses while
    # the cascade keeps deleting the dead ones, every enrichment run recreated
    # them — manual hygiene could never converge. Verify every referenced id
    # in one batch and drop the dead ones from both anchors and entity sets.
    referenced: set[str] = set()
    for indices in cluster_list:
        for idx in indices:
            referenced.add(tagged_fibers[idx].anchor_neuron_id)
            referenced |= tagged_fibers[idx].neuron_ids
    alive = set((await storage.get_neurons_batch(list(referenced))).keys())

    # Build cluster entity sets and anchor IDs
    cluster_entities: list[set[str]] = []
    cluster_anchors: list[str | None] = []
    for indices in cluster_list:
        entity_ids: set[str] = set()
        for idx in indices:
            entity_ids |= tagged_fibers[idx].neuron_ids
        cluster_entities.append(entity_ids & alive)
        # Use the highest-salience fiber whose anchor still exists; a cluster
        # whose anchors are all dead contributes no links.
        anchor: str | None = None
        for candidate in sorted(
            (tagged_fibers[i] for i in indices),
            key=lambda f: f.salience,
            reverse=True,
        ):
            if candidate.anchor_neuron_id in alive:
                anchor = candidate.anchor_neuron_id
                break
        cluster_anchors.append(anchor)

    # Check existing synapses between cluster anchors
    existing_synapses = await storage.get_synapses_paged(type=SynapseType.RELATED_TO)
    existing_pairs: set[tuple[str, str]] = set()
    for syn in existing_synapses:
        existing_pairs.add((syn.source_id, syn.target_id))
        existing_pairs.add((syn.target_id, syn.source_id))

    new_synapses: list[Synapse] = []
    num_clusters = len(cluster_list)
    for i in range(num_clusters):
        if len(new_synapses) >= max_synapses:
            break
        for j in range(i + 1, num_clusters):
            if len(new_synapses) >= max_synapses:
                break
            shared = cluster_entities[i] & cluster_entities[j]
            if not shared:
                continue

            anchor_a = cluster_anchors[i]
            anchor_b = cluster_anchors[j]
            if anchor_a is None or anchor_b is None:
                continue
            if anchor_a == anchor_b:
                continue
            if (anchor_a, anchor_b) in existing_pairs:
                continue

            synapse = Synapse.create(
                source_id=anchor_a,
                target_id=anchor_b,
                type=SynapseType.RELATED_TO,
                weight=0.3,
                metadata={
                    "_enriched": True,
                    "_cross_cluster": True,
                    "_shared_neurons": len(shared),
                },
            )
            new_synapses.append(synapse)
            existing_pairs.add((anchor_a, anchor_b))

    return new_synapses


async def enrich(
    storage: NeuralStorage,
    max_depth: int = 2,
    max_synapses: int = 50,
) -> EnrichmentResult:
    """Run full enrichment: transitive closure + cross-cluster linking.

    Args:
        storage: Storage backend
        max_depth: Maximum transitive chain depth
        max_synapses: Maximum new synapses per strategy

    Returns:
        EnrichmentResult with created synapses
    """
    transitive = await find_transitive_closures(storage, max_depth, max_synapses)
    cross_cluster = await find_cross_cluster_links(storage, max_synapses=max_synapses)

    return EnrichmentResult(
        transitive_synapses=tuple(transitive),
        cross_cluster_synapses=tuple(cross_cluster),
    )
