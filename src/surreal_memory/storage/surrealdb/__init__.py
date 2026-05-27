"""SurrealDB storage backend for Surreal-Memory.

Provides a multi-model storage backend using SurrealDB's graph, document,
and vector search capabilities in a single database.

Usage:
    from surreal_memory.storage.surrealdb import SurrealDBStorage

    storage = SurrealDBStorage(
        url="http://localhost:8001",
        namespace="surreal_memory",
        database="default",
        user="root",
        password="root",
    )
    await storage.initialize()
    storage.set_brain("my-brain")
"""

from surreal_memory.storage.surrealdb.store import SurrealDBStorage

__all__ = ["SurrealDBStorage"]
