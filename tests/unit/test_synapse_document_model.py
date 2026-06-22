"""Regression tests for the synapse document model.

Synapse edges are persisted as document fields (source_id/target_id) on the
synapse table; the redundant write-only connects_to RELATE has been removed and
the source/target indexes now cover the columns lookups actually use.
See Discussion #15 (option A).
"""

from __future__ import annotations

import re

from surreal_memory.storage.surrealdb.schema import SCHEMA_SQL

# Collapse runs of horizontal whitespace so assertions do not depend on the
# column alignment used in the schema source.
_NORM = re.sub(r"[ \t]+", " ", SCHEMA_SQL)


class TestSynapseDocumentModel:
    def test_source_target_fields_declared(self) -> None:
        assert "DEFINE FIELD source_id ON synapse TYPE string;" in _NORM
        assert "DEFINE FIELD target_id ON synapse TYPE string;" in _NORM

    def test_indexes_cover_source_and_target(self) -> None:
        assert "DEFINE INDEX idx_synapse_source ON synapse FIELDS brain_id, source_id;" in _NORM
        assert "DEFINE INDEX idx_synapse_target ON synapse FIELDS brain_id, target_id;" in _NORM

    def test_indexes_drop_unpopulated_out_in(self) -> None:
        # out/in were never written (synapse is not a RELATION table), so the old
        # indexes covered empty columns and every source/target lookup full-scanned.
        assert "idx_synapse_source ON synapse FIELDS brain_id, out" not in _NORM
        assert "idx_synapse_target ON synapse FIELDS brain_id, in" not in _NORM

    def test_connects_to_table_removed(self) -> None:
        # connects_to was write-only and the RELATE never parsed, so nothing read it.
        assert "connects_to" not in SCHEMA_SQL
