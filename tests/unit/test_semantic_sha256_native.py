"""Regression tests for optional native compression of resumable SHA-256 state."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from surreal_memory.engine import _sha256_native
from surreal_memory.engine import semantic_discovery as semantic_discovery_module

_INITIAL_WORDS = list(semantic_discovery_module._SerializableSHA256._INITIAL)
_PORTABLE_COMPRESS = semantic_discovery_module._SerializableSHA256._compress_words


@contextmanager
def _force_native(enabled: bool) -> Iterator[None]:
    """Select native compression or its portable fallback for one test section."""
    prior = _sha256_native._NATIVE_BACKEND_CACHE
    _sha256_native._NATIVE_BACKEND_CACHE = _sha256_native._UNINITIALIZED if enabled else None
    try:
        yield
    finally:
        _sha256_native._NATIVE_BACKEND_CACHE = prior


def _feed(digest: semantic_discovery_module._SerializableSHA256, payload: bytes) -> None:
    offset = 0
    for size in (1, 7, 55, 1, 64, 3, 127, 256, 1024):
        if offset == len(payload):
            break
        chunk = payload[offset : offset + size]
        digest.update(chunk)
        offset += len(chunk)
        yield_state = digest.export()
        assert set(yield_state) == {"h", "n", "b"}
        assert isinstance(yield_state["h"], list)
        assert len(yield_state["h"]) == 8
        assert type(yield_state["n"]) is int
        assert isinstance(yield_state["b"], str)
        assert len(bytes.fromhex(yield_state["b"])) < 64

    if offset < len(payload):
        digest.update(payload[offset:])


@pytest.mark.parametrize("size", [0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 1024, 8193])
def test_digest_and_serialized_state_match_hashlib(size: int) -> None:
    payload = random.Random(size + 9341).randbytes(size)
    digest = semantic_discovery_module._SerializableSHA256()
    portable = semantic_discovery_module._SerializableSHA256()
    reference = hashlib.sha256()

    offset = 0
    for chunk_size in (1, 55, 56, 63, 64, 65, 127, 128, 257, 1024):
        if offset == size:
            break
        chunk = payload[offset : offset + chunk_size]
        with _force_native(True):
            digest.update(chunk)
        with _force_native(False):
            portable.update(chunk)
        reference.update(chunk)
        offset += len(chunk)
        restored = semantic_discovery_module._SerializableSHA256(digest.export())
        assert digest.export() == portable.export()
        assert restored.export() == digest.export()
        assert digest.hexdigest() == reference.hexdigest()
    if offset < size:
        tail = payload[offset:]
        with _force_native(True):
            digest.update(tail)
        with _force_native(False):
            portable.update(tail)
        reference.update(tail)

    assert digest.export() == portable.export()
    assert digest.hexdigest() == hashlib.sha256(payload).hexdigest()
    assert digest.export()["n"] == size
    assert len(bytes.fromhex(digest.export()["b"])) == size % 64


def test_native_and_portable_states_cross_restore_at_block_boundaries() -> None:
    if _sha256_native._get_verified_native_backend(_PORTABLE_COMPRESS) is None:
        pytest.skip("no compatible OpenSSL SHA-256 backend is available")

    payload = random.Random(810223).randbytes(8193)
    split_points = (0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 4095, 8192)
    for split in split_points:
        for source_native, resume_native in ((False, True), (True, False)):
            source = semantic_discovery_module._SerializableSHA256()
            with _force_native(source_native):
                _feed(source, payload[:split])
            checkpoint = source.export()

            resumed = semantic_discovery_module._SerializableSHA256(checkpoint)
            reference = semantic_discovery_module._SerializableSHA256(checkpoint)
            offset = split
            with _force_native(resume_native):
                for chunk_size in (1, 55, 56, 63, 64, 65, 127, 257, 1024):
                    if offset == len(payload):
                        break
                    chunk = payload[offset : offset + chunk_size]
                    resumed.update(chunk)
                    with _force_native(False):
                        reference.update(chunk)
                    assert resumed.export() == reference.export()
                    assert (
                        resumed.hexdigest()
                        == hashlib.sha256(payload[: offset + len(chunk)]).hexdigest()
                    )
                    offset += len(chunk)
                if offset < len(payload):
                    tail = payload[offset:]
                    resumed.update(tail)
                    with _force_native(False):
                        reference.update(tail)
                    assert resumed.export() == reference.export()
            assert resumed.hexdigest() == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "state",
    [
        {"h": [], "n": 0, "b": ""},
        {"h": [0] * 8, "n": 1, "b": ""},
        {"h": [0] * 8, "n": 1, "b": "0"},
        {"h": [0] * 8, "n": 64, "b": "00" * 64},
    ],
)
def test_malformed_serialized_states_keep_rejecting(state: dict[str, object]) -> None:
    with pytest.raises(RuntimeError, match="serialized SHA-256"):
        semantic_discovery_module._SerializableSHA256(state)  # type: ignore[arg-type]


def test_copy_is_independent_after_native_compression() -> None:
    if _sha256_native._get_verified_native_backend(_PORTABLE_COMPRESS) is None:
        pytest.skip("no compatible OpenSSL SHA-256 backend is available")

    left = semantic_discovery_module._SerializableSHA256()
    left.update(b"a" * 129)
    right = left.copy()
    right.update(b"right")
    left.update(b"left")
    assert left.hexdigest() == hashlib.sha256(b"a" * 129 + b"left").hexdigest()
    assert right.hexdigest() == hashlib.sha256(b"a" * 129 + b"right").hexdigest()
    assert left.export() != right.export()


def test_native_backend_unavailable_uses_portable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_sha256_native, "_NATIVE_BACKEND_CACHE", _sha256_native._UNINITIALIZED)
    monkeypatch.setattr(_sha256_native, "_load_openssl_backend", lambda: None)

    payload = random.Random(421).randbytes(1025)
    digest = semantic_discovery_module._SerializableSHA256()
    _feed(digest, payload)
    assert digest.hexdigest() == hashlib.sha256(payload).hexdigest()
    assert _sha256_native._NATIVE_BACKEND_CACHE is None


def test_native_self_test_failure_disables_backend_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongBackend:
        def compress(self, _words: list[int], _blocks: bytes) -> list[int]:
            return _INITIAL_WORDS.copy()

    monkeypatch.setattr(_sha256_native, "_NATIVE_BACKEND_CACHE", _sha256_native._UNINITIALIZED)
    monkeypatch.setattr(_sha256_native, "_load_openssl_backend", lambda: WrongBackend())

    payload = random.Random(771).randbytes(1025)
    digest = semantic_discovery_module._SerializableSHA256()
    _feed(digest, payload)
    assert digest.hexdigest() == hashlib.sha256(payload).hexdigest()
    assert _sha256_native._NATIVE_BACKEND_CACHE is None


def test_native_compression_exception_after_probe_uses_exact_portable_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RaisesAfterSuccessfulProbe:
        calls = 0

        def compress(self, words: list[int], blocks: bytes) -> list[int]:
            self.calls += 1
            if self.calls == 1:
                return _sha256_native._compress_portably(words, blocks, _PORTABLE_COMPRESS)
            raise RuntimeError("injected native compression failure")

    backend = RaisesAfterSuccessfulProbe()
    monkeypatch.setattr(_sha256_native, "_NATIVE_BACKEND_CACHE", _sha256_native._UNINITIALIZED)
    monkeypatch.setattr(_sha256_native, "_load_openssl_backend", lambda: backend)

    payload = random.Random(39041).randbytes(4097)
    digest = semantic_discovery_module._SerializableSHA256()
    digest.update(payload)
    with _force_native(False):
        expected = semantic_discovery_module._SerializableSHA256()
        expected.update(payload)

    assert backend.calls == 2
    assert _sha256_native._NATIVE_BACKEND_CACHE is None
    assert digest.export() == expected.export()
    assert digest.hexdigest() == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "failure", ["unsupported_layout", "missing_library", "load_error", "missing_symbol"]
)
def test_unsupported_or_missing_openssl_backend_uses_portable_fallback(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    monkeypatch.setattr(_sha256_native, "_NATIVE_BACKEND_CACHE", _sha256_native._UNINITIALIZED)

    if failure == "unsupported_layout":
        monkeypatch.setattr(_sha256_native, "_layout_is_supported", lambda: False)
        monkeypatch.setattr(
            _sha256_native.ctypes.util,
            "find_library",
            lambda _name: pytest.fail("library lookup must not run for an unsupported layout"),
        )
    elif failure == "missing_library":
        monkeypatch.setattr(_sha256_native.ctypes.util, "find_library", lambda _name: None)
    elif failure == "load_error":
        monkeypatch.setattr(
            _sha256_native.ctypes.util, "find_library", lambda _name: "libcrypto-test"
        )

        def fail_to_load(_name: str) -> object:
            raise OSError("injected missing library")

        monkeypatch.setattr(_sha256_native.ctypes, "CDLL", fail_to_load)
    else:
        monkeypatch.setattr(
            _sha256_native.ctypes.util, "find_library", lambda _name: "libcrypto-test"
        )
        monkeypatch.setattr(_sha256_native.ctypes, "CDLL", lambda _name: object())

    payload = random.Random(482).randbytes(1025)
    digest = semantic_discovery_module._SerializableSHA256()
    digest.update(payload)
    with _force_native(False):
        expected = semantic_discovery_module._SerializableSHA256()
        expected.update(payload)

    assert _sha256_native._NATIVE_BACKEND_CACHE is None
    assert digest.export() == expected.export()
    assert digest.hexdigest() == hashlib.sha256(payload).hexdigest()
