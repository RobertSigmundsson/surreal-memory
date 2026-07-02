"""Ruflo vault.ts AES-256-GCM content decryption (RFE1 wire format).

Uruboros overlay: neurons written by the Ruflo swarm may carry content
encrypted by vault.ts (magic ``RFE1``). This module mirrors that format so
the engine can read such rows transparently in ``_row_to_neuron``.

Distinct from the upstream Fernet layer (safety/encryption.py), which is a
different format applied at the MCP handler layer — the two do not overlap.

Wire format: ``base64( magic(4) + iv(12) + ciphertext(N) + tag(16) )``.
Key source: ``URUBOROS_ENCRYPTION_KEY`` env (base64 or hex), or the file
``$URUBOROS_SECRETS_DIR/encryption_key`` (default ``~/repos/github/uruboros/.secrets``).

Decryption degrades gracefully: missing key, missing ``cryptography``
package, or non-encrypted content all return the input unchanged.
"""

from __future__ import annotations

import base64
import os

AESGCM: type | None
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover — optional [encryption] extra
    AESGCM = None

MAGIC = b"RFE1"
MAGIC_LEN = 4
IV_LEN = 12
TAG_LEN = 16
KEY_LEN = 32

_ENCRYPTION_KEY: bytes | None = None


def _load_encryption_key() -> bytes | None:
    """Load URUBOROS_ENCRYPTION_KEY from environment or .secrets file."""
    global _ENCRYPTION_KEY
    if _ENCRYPTION_KEY is not None:
        return _ENCRYPTION_KEY if len(_ENCRYPTION_KEY) == KEY_LEN else None

    key_str = os.environ.get("URUBOROS_ENCRYPTION_KEY")
    if not key_str:
        secrets_file = (
            os.path.expanduser(
                os.environ.get(
                    "URUBOROS_SECRETS_DIR",
                    "~/repos/github/uruboros/.secrets",
                )
            )
            + "/encryption_key"
        )
        try:
            with open(secrets_file, "rb") as f:
                key_str = f.read().decode().strip()
        except FileNotFoundError:
            return None

    try:
        key_bytes = base64.b64decode(key_str)
    except Exception:
        # Try as raw bytes (hex encoded)
        try:
            key_bytes = bytes.fromhex(key_str)
        except Exception:
            return None

    if len(key_bytes) != KEY_LEN:
        # Pad or truncate to 32 bytes
        key_bytes = key_bytes[:KEY_LEN].ljust(KEY_LEN, b"\x00")

    _ENCRYPTION_KEY = key_bytes
    return key_bytes


def is_encrypted(content: str) -> bool:
    """Check if content is an encrypted blob (base64-encoded, RFE1 magic)."""
    try:
        raw = base64.b64decode(content)
        return raw[:MAGIC_LEN] == MAGIC
    except Exception:
        return False


def decrypt_content(content: str) -> str:
    """Decrypt content encrypted by Ruflo vault.ts (AES-256-GCM, RFE1 magic).

    Returns original content if the key or the cryptography package is not
    available, or the content is not encrypted.
    """
    if AESGCM is None:
        return content
    key = _load_encryption_key()
    if not key:
        return content

    try:
        raw = base64.b64decode(content)
        if raw[:MAGIC_LEN] != MAGIC:
            return content  # not encrypted

        iv = raw[MAGIC_LEN : MAGIC_LEN + IV_LEN]
        tag = raw[-TAG_LEN:]
        ciphertext = raw[MAGIC_LEN + IV_LEN : -TAG_LEN]

        aesgcm = AESGCM(key)
        plaintext = bytes(aesgcm.decrypt(iv, ciphertext + tag, None))
        return plaintext.decode("utf-8")
    except Exception:
        # Decryption failed — return as-is (plaintext or corrupted)
        return content
