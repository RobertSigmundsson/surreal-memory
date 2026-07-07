# RUN-005 — synapse → native RELATE edges + SurrealDB 3.2.0 gate (surreal-memory v2.6.0)
# BDD user stories = source of truth for the Stage-2 testers (RUNBOOK §8.4).
# Two families: TEST stories (migration correctness / gate) and DAILY-USE stories
# (post-migration product behaviour). Each scenario is tagged with the unit(s) that
# must make it pass and the verification lane (@unit / @integration / @api / @browser).

Feature: Migrate the synapse graph to native RELATE edges under a hard SurrealDB >= 3.2.0 gate
  As the surreal-memory maintainer (Toni)
  I want the synapse table converted to native RELATE edges automatically on upgrade
  So that SurrealDB 3.2.0 graph traversal (GQL MATCH, pushdown) works while every
  existing edge id, fiber, change_log entry and Merkle root stays byte-identical.

  # ---------------------------------------------------------------- TEST STORIES

  @test @U2 @U3 @U6 @integration
  Scenario: Upgrading an existing v7 volume preserves edge count, ids and Merkle root
    Given a SurrealDB v7 database seeded with neurons and flat "synapse" rows in brain "default"
    And a snapshot export plus Merkle root captured before upgrade
    When the store connects for the first time on SurrealDB 3.2.0 and runs apply_migrations
    Then the synapse table is a native RELATION (TYPE RELATION IN neuron OUT neuron)
    And the number of edges equals the pre-upgrade synapse count minus skipped
    And every pre-upgrade synapse id is still resolvable
    And fiber.synapse_ids still resolve to real edges
    And a fresh export is deep-equal to the pre-upgrade export
    And the Merkle root is identical to the pre-upgrade Merkle root

  @test @U2 @integration
  Scenario: A fresh database is created directly at schema v8
    Given an empty SurrealDB 3.2.0 database with no synapse table
    When the store initializes
    Then the synapse table is created directly as a native RELATION
    And schema_meta:version is stamped to 8
    And no migration backup table is created

  @test @U4 @integration
  Scenario: Connecting to a SurrealDB older than 3.2.0 is rejected with a readable hint
    Given a running SurrealDB server at version 3.1.1
    When the store calls initialize()
    Then a StorageVersionError is raised
    And the error message names the minimum required version 3.2.0 and how to upgrade
    And "smem doctor" reports the SurrealDB version check as FAIL with a fix hint

  @test @U2 @integration
  Scenario: A crash mid-migration auto-resumes on the next connect
    Given a v7 database whose migration was interrupted during the converting phase
    And schema_meta:migration_state records the interrupted phase and cursor
    When the store initializes again
    Then the migration resumes from the recorded phase without duplicating edges
    And the final edge count matches the source data
    And schema_meta:version is stamped to 8 only after verification succeeds

  @test @U2 @integration
  Scenario: Two concurrent initialize() calls run exactly one migration
    Given a v7 database and two store instances connecting at the same time
    When both call apply_migrations concurrently
    Then exactly one migration executes under the schema_meta:migration_lock
    And the other waits (or steals the lock only after the 10-minute crash window)
    And the final graph is migrated exactly once with no duplicated edges

  @test @U2 @U3 @U6 @integration
  Scenario: Self-loops, orphan edges and dashed ids survive the migration
    Given a v7 database containing a self-loop edge, an orphan edge, and an id with a dash
    When the migration runs
    Then the self-loop edge exists as a native RELATE edge (or is skipped-and-logged, never crashing)
    And the orphan edge is preserved (schema is not ENFORCED)
    And the dashed id is normalised consistently and remains resolvable

  # ------------------------------------------------------------ DAILY-USE STORIES

  @daily @U3 @U6 @integration @api
  Scenario: Recall works after migration
    Given a migrated v8 database with existing memories
    When the user issues a recall query
    Then relevant memories are returned using the native-edge queries
    And no error is raised about the synapse model

  @daily @U3 @U5 @U6 @integration
  Scenario: Path search returns paths with GQL fast-path or BFS fallback
    Given a migrated v8 database with a multi-hop chain of neurons
    When the user requests a path between two connected neurons
    Then a valid path is returned
    And when eval::gql is available the GQL MATCH SHORTEST fast-path is used
    And when GQL raises or is unavailable the BFS fallback returns the same path

  @daily @U8 @browser
  Scenario: The dashboard graph view renders the migrated graph
    Given a migrated v8 database served by the MCP/dashboard API
    When the user opens the dashboard graph view
    Then the migrated neurons and edges are rendered without error

  @daily @U4 @api
  Scenario: smem doctor reports version OK and migration done
    Given a migrated v8 database on SurrealDB 3.2.0
    When the user runs "smem doctor"
    Then the SurrealDB version check reports OK (>= 3.2.0)
    And "smem doctor --synapse-migration status" reports the migration as done

  @daily @U3 @U6 @integration
  Scenario: Export/import round-trip stays byte-stable after migration
    Given a migrated v8 database
    When the graph is exported and re-imported
    Then the round-trip is byte-stable
    And the Merkle root is unchanged
