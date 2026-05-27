"""Tests for auto-redact secrets (Phase F)."""

from __future__ import annotations

from surreal_memory.safety.sensitive import (
    auto_redact_content,
)
from surreal_memory.unified_config import SafetyConfig


class TestAutoRedactContent:
    def test_no_sensitive_content_passthrough(self) -> None:
        content = "We decided to use FastAPI for the REST server"
        redacted, matches, content_hash = auto_redact_content(content)
        assert redacted == content
        assert matches == []
        assert content_hash is None

    def test_api_key_redacted_at_severity_3(self) -> None:
        content = "The api_key = sk-1234567890abcdef1234567890"
        redacted, matches, content_hash = auto_redact_content(content, min_severity=3)
        assert "[REDACTED]" in redacted
        assert len(matches) >= 1
        assert content_hash is not None
        # Original content should not appear
        assert "sk-1234567890abcdef1234567890" not in redacted

    def test_standalone_jwt_not_redacted_at_severity_3(self) -> None:
        """A bare JWT (no 'token =' prefix) is severity 2, not auto-redacted at 3."""
        # Use a bare JWT without the "token = " prefix that triggers Generic Token pattern
        content = "Found this in logs: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        redacted, matches, content_hash = auto_redact_content(content, min_severity=3)
        # Bare JWT is severity 2 only, should NOT be redacted at min_severity=3
        sev3_matches = [m for m in matches if m.severity >= 3]
        assert sev3_matches == []

    def test_jwt_redacted_at_severity_2(self) -> None:
        """JWT should be redacted when min_severity is lowered to 2."""
        content = "token = eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        redacted, matches, content_hash = auto_redact_content(content, min_severity=2)
        assert len(matches) >= 1
        assert content_hash is not None

    def test_password_redacted(self) -> None:
        content = "password = MySecretPassword123!"
        redacted, matches, _ = auto_redact_content(content, min_severity=3)
        assert "[REDACTED]" in redacted
        assert "MySecretPassword123!" not in redacted

    def test_content_hash_is_sha256(self) -> None:
        content = "api_key = sk-1234567890abcdef1234567890"
        _, _, content_hash = auto_redact_content(content, min_severity=3)
        assert content_hash is not None
        assert len(content_hash) == 64  # SHA-256 hex digest

    def test_multiple_secrets_all_redacted(self) -> None:
        content = (
            "api_key = sk-1234567890abcdef1234567890\nsecret = MyVeryLongSecretValue1234567890\n"
        )
        redacted, matches, _ = auto_redact_content(content, min_severity=3)
        assert len(matches) >= 2
        assert "sk-1234567890abcdef1234567890" not in redacted


class TestSafetyConfig:
    def test_default_severity_3(self) -> None:
        cfg = SafetyConfig()
        assert cfg.auto_redact_min_severity == 3

    def test_from_dict_clamped(self) -> None:
        cfg = SafetyConfig.from_dict({"auto_redact_min_severity": 5})
        assert cfg.auto_redact_min_severity == 3  # clamped to max 3

    def test_from_dict_min_clamped(self) -> None:
        cfg = SafetyConfig.from_dict({"auto_redact_min_severity": 0})
        assert cfg.auto_redact_min_severity == 1  # clamped to min 1

    def test_to_dict_roundtrip(self) -> None:
        cfg = SafetyConfig(auto_redact_min_severity=2)
        d = cfg.to_dict()
        restored = SafetyConfig.from_dict(d)
        assert restored.auto_redact_min_severity == 2
