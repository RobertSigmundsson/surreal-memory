"""Optional OpenSSL acceleration for serializable SHA-256 block compression.

The checkpoint format stores the eight chaining words and the uncompressed tail,
so it cannot use ``hashlib``'s opaque context. This module uses OpenSSL's public
low-level SHA256 context only after checking its documented layout and comparing
its compression result with the portable implementation supplied by the caller.
Every unsupported or failed native path falls back to that portable implementation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
from collections.abc import Callable
from typing import Protocol, cast

_BLOCK_SIZE = 64
_WORD_MASK = 0xFFFFFFFF
_INITIAL_WORDS = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)

PortableCompressor = Callable[[list[int], bytes], list[int]]


class _SHA256Context(ctypes.Structure):
    """The public OpenSSL SHA256_CTX layout from openssl/sha.h."""

    _fields_ = [
        ("h", ctypes.c_uint * 8),
        ("nl", ctypes.c_uint),
        ("nh", ctypes.c_uint),
        ("data", ctypes.c_uint * 16),
        ("num", ctypes.c_uint),
        ("md_len", ctypes.c_uint),
    ]


class _NativeCompressor(Protocol):
    def compress(self, words: list[int], blocks: bytes) -> list[int]: ...


class _OpenSSLCompressor:
    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        context_pointer = ctypes.POINTER(_SHA256Context)
        self._init = library.SHA256_Init
        self._init.argtypes = [context_pointer]
        self._init.restype = ctypes.c_int
        self._update = library.SHA256_Update
        self._update.argtypes = [context_pointer, ctypes.c_void_p, ctypes.c_size_t]
        self._update.restype = ctypes.c_int

    def compress(self, words: list[int], blocks: bytes) -> list[int]:
        if len(words) != 8 or any(word < 0 or word > _WORD_MASK for word in words):
            raise ValueError("invalid SHA-256 chaining words")
        if not blocks or len(blocks) % _BLOCK_SIZE:
            raise ValueError("native SHA-256 input must contain complete blocks")

        context = _SHA256Context()
        pointer = ctypes.byref(context)
        if self._init(pointer) != 1:
            raise RuntimeError("OpenSSL SHA256_Init failed")
        if (
            list(context.h) != list(_INITIAL_WORDS)
            or context.nl != 0
            or context.nh != 0
            or context.num != 0
            or context.md_len != 32
        ):
            raise RuntimeError("OpenSSL SHA256_CTX initialization did not match its ABI")

        for index, word in enumerate(words):
            context.h[index] = word
        context.nl = 0
        context.nh = 0
        context.num = 0
        context.md_len = 32
        for index in range(16):
            context.data[index] = 0

        bit_count = len(blocks) * 8
        if self._update(pointer, ctypes.c_char_p(blocks), len(blocks)) != 1:
            raise RuntimeError("OpenSSL SHA256_Update failed")
        if (
            context.num != 0
            or context.nl != (bit_count & _WORD_MASK)
            or context.nh != ((bit_count >> 32) & _WORD_MASK)
        ):
            raise RuntimeError("OpenSSL SHA256_CTX buffer or byte count was incompatible")
        return list(context.h)


_UNINITIALIZED = object()
_NATIVE_BACKEND_CACHE: _NativeCompressor | object | None = _UNINITIALIZED
_NATIVE_BACKEND_LOCK = threading.Lock()


def _layout_is_supported() -> bool:
    return (
        ctypes.sizeof(ctypes.c_uint) == 4
        and ctypes.sizeof(_SHA256Context) == 112
        and _SHA256Context.h.offset == 0
        and _SHA256Context.nl.offset == 32
        and _SHA256Context.nh.offset == 36
        and _SHA256Context.data.offset == 40
        and _SHA256Context.num.offset == 104
        and _SHA256Context.md_len.offset == 108
    )


def _load_openssl_backend() -> _NativeCompressor | None:
    if not _layout_is_supported():
        return None
    library_name = ctypes.util.find_library("crypto")
    if not library_name:
        return None
    try:
        return _OpenSSLCompressor(ctypes.CDLL(library_name))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _compress_portably(
    words: list[int], blocks: bytes, portable_compress: PortableCompressor
) -> list[int]:
    result = words.copy()
    for offset in range(0, len(blocks), _BLOCK_SIZE):
        result = portable_compress(result, blocks[offset : offset + _BLOCK_SIZE])
    return result


def _get_verified_native_backend(
    portable_compress: PortableCompressor,
) -> _NativeCompressor | None:
    global _NATIVE_BACKEND_CACHE
    if _NATIVE_BACKEND_CACHE is _UNINITIALIZED:
        with _NATIVE_BACKEND_LOCK:
            if _NATIVE_BACKEND_CACHE is _UNINITIALIZED:
                backend: _NativeCompressor | None = None
                try:
                    candidate = _load_openssl_backend()
                    if candidate is not None:
                        probe_blocks = bytes((index * 73 + 19) & 0xFF for index in range(3 * 64))
                        expected = _compress_portably(
                            list(_INITIAL_WORDS), probe_blocks, portable_compress
                        )
                        actual = candidate.compress(list(_INITIAL_WORDS), probe_blocks)
                        if actual == expected:
                            backend = candidate
                except Exception:
                    backend = None
                _NATIVE_BACKEND_CACHE = backend

    cached = _NATIVE_BACKEND_CACHE
    if cached is _UNINITIALIZED or cached is None:
        return None
    return cast("_NativeCompressor | None", cached)


def compress_sha256_blocks(
    words: list[int], blocks: bytes, portable_compress: PortableCompressor
) -> list[int]:
    """Compress complete blocks, preferring a verified OpenSSL backend."""
    if not blocks:
        return words.copy()
    if len(blocks) % _BLOCK_SIZE:
        raise ValueError("SHA-256 compression input must contain complete blocks")

    backend = _get_verified_native_backend(portable_compress)
    if backend is not None:
        try:
            return backend.compress(words, blocks)
        except Exception:
            global _NATIVE_BACKEND_CACHE
            with _NATIVE_BACKEND_LOCK:
                _NATIVE_BACKEND_CACHE = None
    return _compress_portably(words, blocks, portable_compress)
