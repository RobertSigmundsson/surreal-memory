"""Prompt templates for LLM-based dedup judgment (Tier 3)."""

from __future__ import annotations

DEDUP_SYSTEM_PROMPT = """\
You are a deduplication judge for a neural memory system. Your job is to determine \
whether two memory entries are semantically equivalent (duplicates) or distinct memories.

Respond with EXACTLY one of:
- DUPLICATE: The entries convey the same core information
- DISTINCT: The entries convey meaningfully different information
- UNCERTAIN: You cannot confidently determine

Follow your answer with a brief reason on the next line."""

DEDUP_USER_PROMPT = """\
Memory A:
{content_a}

Memory B:
{content_b}

Are these memories duplicates or distinct? Consider:
1. Do they convey the same core fact/decision/instruction?
2. Is one a more specific version of the other?
3. Would keeping both add redundant information?"""


def _escape_braces(text: str) -> str:
    """Escape curly braces in user content to prevent format injection."""
    return text.replace("{", "{{").replace("}", "}}")


def format_dedup_prompt(content_a: str, content_b: str) -> str:
    """Format the user prompt for dedup judgment.

    Content is escaped to prevent .format() injection and truncated to 500 chars.
    """
    safe_a = _escape_braces(content_a[:500])
    safe_b = _escape_braces(content_b[:500])
    return DEDUP_USER_PROMPT.format(
        content_a=safe_a,
        content_b=safe_b,
    )
