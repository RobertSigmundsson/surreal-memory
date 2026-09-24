"""Explicit memory write shared by ``smem remember`` (CLI) and the recall-http shim (``/v1/remember``).

Moved out of ``cli/commands/memory.py`` so both callers run the SAME write (sensitive-content gate,
type, default expiry, priority, encoder with dedup, ephemeral flag, ``typed_memory``) — one code
path, so a pod's write and a host CLI write of the same text produce the same graph. Kept in
``engine`` (not ``cli``): importing anything under ``cli`` pulls in the whole Typer app.

The only difference between callers is :class:`Attribution`: the CLI keeps ``source="user_input"``
and no ``stored_by``; the shim records the pod (``stored_by`` in encoder metadata lands on the anchor
neuron and the fiber; ``typed_memory.source`` = tor; provenance ``created_by`` = agent id). None of it
adds neurons or synapses.

No background work: everything runs inline in the caller's task.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

from surreal_memory.core.memory_types import (
    DEFAULT_EXPIRY_DAYS,
    MemoryType,
    Priority,
    TypedMemory,
    suggest_memory_type,
)
from surreal_memory.engine.dedup.factory import build_dedup_pipeline
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.safety.sensitive import (
    SensitiveMatch,
    check_sensitive_content,
    filter_sensitive_content,
)
from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from surreal_memory.core.brain import BrainConfig

SOURCE_CLI: Final = "user_input"
SENSITIVE_MIN_SEVERITY: Final = 2


class SensitiveContentError(ValueError):
    """Sensitive content without ``force``/``redact``. ``str(exc)`` never carries the match."""

    def __init__(self, matches: list[SensitiveMatch]) -> None:
        super().__init__(f"sensitive_content:{len(matches)}")
        self.matches = tuple(matches)

    @property
    def types(self) -> list[str]:
        return sorted({m.type.value for m in self.matches})


class InvalidMemoryTypeError(ValueError):
    def __init__(self) -> None:
        super().__init__("invalid_memory_type")
        self.valid = tuple(t.value for t in MemoryType)


@dataclass(frozen=True)
class ContentCheck:
    content: str  # content to store (redacted when redact=True and something matched)
    matches: tuple[SensitiveMatch, ...]
    redacted: bool


def check_content(content: str, *, force: bool, redact: bool) -> ContentCheck:
    """Sensitive-content gate of ``smem remember`` (severity >= 2)."""
    matches = check_sensitive_content(content, min_severity=SENSITIVE_MIN_SEVERITY)
    if matches and not force and not redact:
        raise SensitiveContentError(matches)
    if redact and matches:
        redacted, _ = filter_sensitive_content(content)
        return ContentCheck(redacted, tuple(matches), True)
    return ContentCheck(content, tuple(matches), False)


def resolve_memory_type(memory_type: str | None, content: str) -> MemoryType:
    """Explicit type (case-insensitive) or auto-detected from the content."""
    if memory_type:
        try:
            return MemoryType(memory_type.lower())
        except ValueError:
            raise InvalidMemoryTypeError() from None
    return suggest_memory_type(content)


def resolve_expiry_days(
    mem_type: MemoryType, expires: int | None, *, ephemeral: bool
) -> int | None:
    """Explicit expiry wins; else the per-type default; ephemeral without one gets 1 day."""
    expiry_days = expires if expires is not None else DEFAULT_EXPIRY_DAYS.get(mem_type)
    if ephemeral and expiry_days is None:
        expiry_days = 1
    return expiry_days


def resolve_priority(priority: int | None) -> tuple[Priority, bool]:
    """(priority, was it explicit)."""
    if priority is None:
        return Priority.NORMAL, False
    return Priority.from_int(priority), True


@dataclass(frozen=True)
class Attribution:
    source: str = SOURCE_CLI  # -> typed_memory.source
    created_by: str | None = None  # -> typed_memory provenance.created_by (None = default "user")
    stored_by: Mapping[str, str] | None = None  # -> encoder metadata "stored_by" (anchor + fiber)


CLI_ATTRIBUTION: Final = Attribution()


@dataclass(frozen=True)
class StoredMemory:
    fiber_id: str
    anchor_neuron_id: str
    typed_mem: TypedMemory
    neurons_created: int
    neurons_linked: int
    synapses_created: int


async def encode_and_store(
    storage: Any,
    brain_config: BrainConfig,
    content: str,
    *,
    tags: set[str] | None,
    mem_type: MemoryType,
    mem_priority: Priority,
    expiry_days: int | None,
    project_id: str | None,
    event_timestamp: datetime | None = None,
    ephemeral: bool = False,
    priority_was_explicit: bool = False,
    attribution: Attribution = CLI_ATTRIBUTION,
) -> StoredMemory:
    """Encode content into the neural graph and store its typed-memory metadata."""
    encoder = MemoryEncoder(storage, brain_config, dedup_pipeline=build_dedup_pipeline(storage))
    storage.disable_auto_save()
    try:
        # A priority the caller asked for must reach the FIBER, not only typed_memory: retrieval
        # scores fibers. A merely defaulted priority is not written.
        encode_metadata: dict[str, Any] | None = None
        if priority_was_explicit:
            encode_metadata = {"priority": mem_priority.value}
        if attribution.stored_by is not None:
            encode_metadata = {**(encode_metadata or {}), "stored_by": dict(attribution.stored_by)}

        result = await encoder.encode(
            content=content,
            timestamp=event_timestamp or utcnow(),
            metadata=encode_metadata,
            tags=tags,
        )

        if ephemeral:
            ephemeral_ids = [n.id for n in result.neurons_created]
            if ephemeral_ids:
                await storage.update_neurons_ephemeral_batch(ephemeral_ids, ephemeral=True)

        typed_mem = TypedMemory.create(
            fiber_id=result.fiber.id,
            memory_type=mem_type,
            priority=mem_priority,
            source=attribution.source,
            expires_in_days=expiry_days,
            tags=tags,
            project_id=project_id,
        )
        if attribution.created_by is not None:
            typed_mem = dataclasses.replace(
                typed_mem,
                provenance=dataclasses.replace(
                    typed_mem.provenance, created_by=attribution.created_by
                ),
            )
        await storage.add_typed_memory(typed_mem)
        await storage.batch_save()
    finally:
        storage.enable_auto_save()

    return StoredMemory(
        fiber_id=result.fiber.id,
        anchor_neuron_id=result.fiber.anchor_neuron_id,
        typed_mem=typed_mem,
        neurons_created=len(result.neurons_created),
        neurons_linked=len(result.neurons_linked),
        synapses_created=len(result.synapses_created),
    )


def response_dict(
    stored: StoredMemory,
    *,
    content: str,
    mem_type: MemoryType,
    mem_priority: Priority,
    ephemeral: bool,
    project: str | None,
    forced_matches: int,
) -> dict[str, Any]:
    """The ``smem remember --json`` response (keys and order as the CLI has always printed)."""
    response: dict[str, Any] = {
        "message": f"Remembered: {content[:50]}{'...' if len(content) > 50 else ''}",
        "fiber_id": stored.fiber_id,
        "memory_type": mem_type.value,
        "priority": mem_priority.name.lower(),
        "neurons_created": stored.neurons_created,
        "neurons_linked": stored.neurons_linked,
        "synapses_created": stored.synapses_created,
    }
    if ephemeral:
        response["ephemeral"] = True
        response["message"] += " [ephemeral — auto-expires in 24h]"
    if project:
        response["project"] = project
    if stored.typed_mem.expires_at:
        response["expires_in_days"] = stored.typed_mem.days_until_expiry
    if forced_matches:
        response["warnings"] = [
            f"[!] Stored with {forced_matches} sensitive item(s) - consider using --redact"
        ]
    return response
