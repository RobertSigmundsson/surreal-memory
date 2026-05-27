"""Safety utilities for Surreal-Memory.

This module provides tools for:
- Sensitive content detection
- Memory freshness evaluation
- Privacy protection
"""

from surreal_memory.safety.encryption import (
    EncryptionResult,
    MemoryEncryptor,
)
from surreal_memory.safety.freshness import (
    FreshnessLevel,
    evaluate_freshness,
    get_freshness_warning,
)
from surreal_memory.safety.sensitive import (
    SensitiveMatch,
    SensitivePattern,
    check_sensitive_content,
    filter_sensitive_content,
    get_default_patterns,
)

__all__ = [
    "EncryptionResult",
    "FreshnessLevel",
    "MemoryEncryptor",
    "SensitiveMatch",
    "SensitivePattern",
    "check_sensitive_content",
    "evaluate_freshness",
    "filter_sensitive_content",
    "get_default_patterns",
    "get_freshness_warning",
]
