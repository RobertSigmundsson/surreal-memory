"""Parametrized corpus tests for suggest_memory_type.

Locks the 11-of-15-types coverage of the keyword classifier:
    FACT (default), TODO, DECISION, INSIGHT, INSTRUCTION,
    PREFERENCE, WORKFLOW, REFERENCE, BOUNDARY, TOOL, CONTEXT.

ERROR is intentionally NOT auto-detected: it carries a deletion TTL,
so a keyword misfire on durable content (post-mortems, audit reports)
would silently schedule that content for deletion. ERROR only comes
from an explicit `type=` argument.

Edge-case section covers known precedence collisions where naive
ordering would silently misclassify safety-critical content.
"""

from __future__ import annotations

import pytest

from surreal_memory.core.memory_types import MemoryType, suggest_memory_type

# Per-type corpus. Every entry must be at least 10 sentences. Sentences
# should be representative of how a real user / hook produces content.
CLASSIFIER_CORPUS: dict[MemoryType, list[str]] = {
    MemoryType.BOUNDARY: [
        "Never use eval() in production code",
        "Must not commit secrets to the repo",
        "Always ask before running rm -rf",
        "Always confirm before pushing to main",
        "Don't ever disable signature verification",
        "Must never share credentials in chat",
        "Never use innerHTML with untrusted input",
        "Always ask before modifying production data",
        "Must not bypass MFA for admin accounts",
        "Don't ever skip the security review step",
    ],
    MemoryType.TODO: [
        "TODO: refactor the auth module",
        "FIXME: race condition in encoder",
        "Need to add retries to the API client",
        "Have to migrate the legacy endpoints",
        "Remember to update the changelog",
        "Should refactor the auth module next sprint",
        "Must finish the report by Friday",
        "TODO: write integration tests for the new flow",
        "FIXME: leaking file handles on shutdown",
        "Need to escalate the SLA breach",
    ],
    MemoryType.DECISION: [
        "We decided to use PostgreSQL over MySQL",
        "Chose React over Vue for the dashboard",
        "Picked TypeScript for the new service",
        "Selected Stripe as the payment provider",
        "Opted for a monorepo structure",
        "Going with rye for dependency management",
        "Switched to Tailwind for styling",
        "Rejected GraphQL in favor of REST",
        "Went with Postgres trigram for fuzzy search",
        "Chose Redis over Memcached for the cache",
    ],
    MemoryType.INSIGHT: [
        "Learned that the encoder is the bottleneck",
        "Realized the cache hit rate drops on Mondays",
        "Discovered a data leak in the staging logs",
        "Found that the slow path is gated by a global lock",
        "Turns out the migration wasn't idempotent",
        "Lesson learned: always double-write during cutovers",
        "Noticed that the RPS spikes correlate with deploys",
        "Figured out that the issue only fires on UTC midnight",
        "The pattern is that large brains stress the GC",
        "Key insight: synapses dominate retrieval cost at scale",
    ],
    MemoryType.INSTRUCTION: [
        "Always use type hints in Python",
        "Make sure to run tests before commit",
        "Don't forget to bump the version when releasing",
        "Don't forget to regenerate the lockfile",
        "Always use parametrized queries",
        "Make sure the SBOM is up to date",
        "Always use the project virtualenv",
        "Make sure to record the deploy in the changelog",
        "Always use idempotent migrations",
        "Don't forget to seal the secret before commit",
    ],
    MemoryType.PREFERENCE: [
        "I prefer tabs over spaces",
        "User prefers concise answers without preamble",
        "Preferred editor is Neovim",
        "I like vertical splits over horizontal",
        "Favorite linter is ruff",
        "Hate trailing whitespace in diffs",
        "Dislike emojis in commit messages",
        "Prefers async over sync where possible",
        "User prefers ISO date format",
        "Preferred shell is fish",
    ],
    MemoryType.WORKFLOW: [
        "Workflow: branch off main, PR, squash merge",
        "Pipeline runs lint, tests, build, deploy",
        "Deploy to staging first, then production",
        "CI/CD enforces the security scan gate",
        "Release process: tag, draft notes, announce",
        "The standard process is to file an RFC first",
        "Release flow goes through staging then prod",
        "Deploy flow uses blue/green with manual approval",
        "Workflow involves an RFC, design review, then build",
        "Pipeline is gated on a smoke test in staging",
    ],
    MemoryType.REFERENCE: [
        "Docs at https://example.com/api",
        "See the AWS documentation for IAM policies",
        "Reference: https://datatracker.ietf.org/doc/html/rfc7519",
        "Documentation lives in the wiki",
        "https://surrealdb.com/docs/surrealql",
        "API docs are at https://api.example.org/v2",
        "Documentation auto-generated from docstrings",
        "Refer to the official Postgres documentation",
        "https://www.python.org/dev/peps/pep-0008/",
        "Docs explain the failover semantics",
    ],
    MemoryType.TOOL: [
        "The 'rg' command supports PCRE2 with -P",
        "ruff CLI honors the project pyproject.toml",
        "Use the --check flag to dry-run",
        "Invoke the formatter via 'ruff format'",
        "Run with --verbose to debug",
        "The subcommand 'cargo build' produces release binaries",
        "git sub-command 'rebase -i' enables history editing",
        "fd command is a faster replacement for find",
        "Use the cli to inspect the current config",
        "Invoke pytest with --cov to measure coverage",
    ],
    MemoryType.CONTEXT: [
        "Currently working on the SurrealDB fork",
        "Right now we're focused on the migration tooling",
        "This session is about classifier expansion",
        "In this project we use ruff and mypy strictly",
        "For client X, the SLA is 99.95%",
        "Currently working on hardening the schema",
        "Right now the team is in a freeze window",
        "This session: profiling the encoder",
        "In this project, all PRs require two reviewers",
        "For client Acme, deploys happen Monday mornings",
    ],
    MemoryType.FACT: [
        "Python 3.11 was released in October 2022",
        "SurrealDB 2.0 added experimental graph DSL",
        "PostgreSQL 16 introduced logical replication parallelism",
        "ruff is written in Rust and ships as a single binary",
        "GPT-4 has a 128k context window",
        "AWS Lambda has a 15-minute execution limit",
        "Redis 7 introduced multi-key transactions",
        "TypeScript 5.0 stabilized decorators",
        "Kubernetes deprecated PodSecurityPolicy in 1.21",
        "The Rust 2024 edition stabilized async fn in traits",
    ],
}


def _flatten() -> list[tuple[str, MemoryType]]:
    out: list[tuple[str, MemoryType]] = []
    for mtype, samples in CLASSIFIER_CORPUS.items():
        for sentence in samples:
            out.append((sentence, mtype))
    return out


@pytest.mark.parametrize(("content", "expected"), _flatten())
def test_classifier_corpus(content: str, expected: MemoryType) -> None:
    """Each corpus sentence classifies to its expected MemoryType."""
    assert suggest_memory_type(content) == expected, (
        f"Expected {expected.value} for {content!r}, got {suggest_memory_type(content).value}"
    )


def test_corpus_size_at_least_ten_per_type() -> None:
    """Lock the 10-per-type minimum so corpus stays representative."""
    for mtype, samples in CLASSIFIER_CORPUS.items():
        assert len(samples) >= 10, f"{mtype.value} corpus has {len(samples)} samples; need ≥10"


def test_cognitive_only_types_excluded_from_corpus() -> None:
    """HYPOTHESIS, PREDICTION, SCHEMA must not be in the auto-classifier corpus.

    Those types are produced by cognitive_handler / knowledge_gaps flows
    against structured input, not inferred from raw content. Including
    them here would imply the classifier should detect them, which it
    intentionally does not.
    """
    cognitive_only = {MemoryType.HYPOTHESIS, MemoryType.PREDICTION, MemoryType.SCHEMA}
    assert cognitive_only.isdisjoint(CLASSIFIER_CORPUS.keys())


# ---- ERROR is never auto-detected ---------------------------------------

# Error-keyword sentences that historically classified as ERROR. ERROR
# gets a deletion TTL, so auto-assigning it destroys durable content;
# these must now land on TTL-free types instead.
ERROR_KEYWORD_SENTENCES: list[tuple[str, MemoryType]] = [
    ("The API returns a 500 error on empty payload", MemoryType.FACT),
    ("Bug: pagination skips the last page", MemoryType.FACT),
    ("Crash on macOS when opening the export dialog", MemoryType.FACT),
    ("Exception thrown by the parser on malformed JSON", MemoryType.FACT),
    ("Traceback indicates a stale connection pool", MemoryType.FACT),
    ("Build failed because of a missing dependency", MemoryType.FACT),
    ("Encoder crashed with an OOM exception", MemoryType.FACT),
    ("Login flow is broken on Safari iOS", MemoryType.WORKFLOW),
    ("Migration failed mid-run leaving partial state", MemoryType.FACT),
    ("Bug in the rate limiter — leaks tokens on retry", MemoryType.FACT),
]


@pytest.mark.parametrize(("content", "expected"), ERROR_KEYWORD_SENTENCES)
def test_error_keywords_do_not_auto_classify_as_error(content: str, expected: MemoryType) -> None:
    result = suggest_memory_type(content)
    assert result != MemoryType.ERROR, f"auto-detected ERROR for {content!r}"
    assert result == expected


def test_error_type_excluded_from_corpus() -> None:
    """ERROR must not be in the auto-classifier corpus (explicit-only type)."""
    assert MemoryType.ERROR not in CLASSIFIER_CORPUS


def test_classifier_never_emits_error() -> None:
    """No input may auto-classify as ERROR — including report-like prose
    that merely mentions failures (the 2026-08-22 defect: an audit
    report auto-typed `error` and got a 30-day deletion TTL)."""
    report_like = (
        "AUDYT SMEM 2026-08-22: pelny raport z dowodami, bramka danych "
        "przeszla, error handling zweryfikowany (raport)"
    )
    all_sentences = [s for samples in CLASSIFIER_CORPUS.values() for s in samples]
    all_sentences += [s for s, _ in ERROR_KEYWORD_SENTENCES]
    all_sentences.append(report_like)
    offenders = [s for s in all_sentences if suggest_memory_type(s) == MemoryType.ERROR]
    assert offenders == []


# ---- Precedence collision tests -----------------------------------------


class TestPrecedenceCollisions:
    """Targeted tests where two branches could fire and ordering matters."""

    def test_never_use_routes_to_boundary_not_instruction(self) -> None:
        """'never use' is in BOTH BOUNDARY and (historically) INSTRUCTION.

        Safety must win. This is the canonical reason BOUNDARY is
        checked first.
        """
        assert suggest_memory_type("Never use eval() in production") == MemoryType.BOUNDARY
        assert suggest_memory_type("never use innerHTML on user input") == MemoryType.BOUNDARY

    def test_must_not_routes_to_boundary_not_todo(self) -> None:
        """'must' fires TODO; 'must not' fires BOUNDARY. BOUNDARY runs first."""
        assert suggest_memory_type("Must not log credentials") == MemoryType.BOUNDARY

    def test_actionable_should_routes_to_todo(self) -> None:
        """Plain actionable 'should X' (no disqualifier words) is TODO."""
        assert suggest_memory_type("Should refactor the auth module") == MemoryType.TODO

    def test_should_with_because_disqualifier_falls_through(self) -> None:
        """The TODO disqualifier list intentionally drops descriptive 'should'.

        'Should refactor X because Y' contains 'because' which is in the
        disqualifier list, so TODO does not fire — content falls through
        to FACT (no other branch matches).
        """
        assert (
            suggest_memory_type("Should refactor X because the architecture forced it")
            == MemoryType.FACT
        )

    def test_deploy_pipeline_routes_to_workflow_not_tool(self) -> None:
        """'deploy' (WORKFLOW) wins over generic 'command'/'invoke' that
        WORKFLOW content might mention. TOOL runs after WORKFLOW.
        """
        assert (
            suggest_memory_type("Deploy pipeline runs build then push command")
            == MemoryType.WORKFLOW
        )

    def test_currently_working_on_routes_to_context_not_fact(self) -> None:
        """CONTEXT fires before the FACT default."""
        assert suggest_memory_type("Currently working on the SurrealDB fork") == MemoryType.CONTEXT
