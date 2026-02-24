"""MCN-v0.1 ChromaDB Patch Registry — Persistent Vector Store.

Drop-in replacement for PatchRegistry that maintains the same interface:
    search(context_vector, top_k) -> list[PatchEntry]
    register_candidate(...) -> Optional[PatchEntry]
    record_attempt(task_id) -> int
    state_dict() / from_state_dict()
    n_patches -> int

Uses ChromaDB for:
    - ANN (approximate nearest neighbor) search via HNSW index
    - Persistent storage (survives container restarts)
    - Metadata filtering (by exception type, function name, etc.)
    - Cosine distance metric (matching original PatchRegistry)

Patch data is stored in ChromaDB; attempt counts are stored in-memory
(lightweight, persisted via Redis state_dict mechanism).

Usage:
    registry = ChromaPatchRegistry(
        dim=18,
        chroma_url="http://chromadb:8000",
        collection_name="mcn_patches",
    )
    registry.record_attempt("task_123")
    registry.record_attempt("task_123")
    patch = registry.register_candidate(
        task_id="task_123", description="Sort a list...",
        function_name="sort_safe", code="def sort_safe(xs): ...",
        context_vector=ctx, passed=True, failure_category="IndexError",
    )
    matches = registry.search(new_context, top_k=1)
    hint = matches[0].lesson if matches else ""
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray

from mcn.memory import PatchEntry

logger = logging.getLogger(__name__)


class ChromaPatchRegistry:
    """ChromaDB-backed patch store for knowledge diffusion.

    Uses a single ChromaDB collection with:
        - Embeddings: 18-dim context vectors (same as PatchRegistry)
        - Distance: cosine (matching in-memory implementation)
        - Metadata: function_name, failure_category, attempt_count, etc.
        - Documents: patch lesson text (for retrieval)

    Attributes:
        dim: Dimensionality of context vectors.
        min_attempts: Minimum attempts for patch qualification.
        max_patches: Maximum patches in store (soft limit via count check).
        similarity_threshold: Minimum cosine similarity for a match.
    """

    def __init__(
        self,
        dim: int = 18,
        min_attempts: int = 2,
        max_patches: int = 5000,
        similarity_threshold: float = 0.3,
        chroma_url: str = "",
        persist_dir: str = "",
        collection_name: str = "mcn_patches",
    ) -> None:
        self.dim = dim
        self.min_attempts = min_attempts
        self.max_patches = max_patches
        self.similarity_threshold = similarity_threshold

        # Import chromadb lazily to avoid import errors when disabled
        import chromadb

        # Connect to ChromaDB
        if chroma_url:
            parsed = urlparse(chroma_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8000
            self._client = chromadb.HttpClient(host=host, port=port)
            logger.info("ChromaDB connected: %s:%d", host, port)
        elif persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
            logger.info("ChromaDB local persistent: %s", persist_dir)
        else:
            self._client = chromadb.Client()
            logger.info("ChromaDB ephemeral client (in-memory)")

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Attempt counts (in-memory, persisted via state_dict)
        self._attempt_counts: dict[str, int] = {}

        logger.info(
            "ChromaPatchRegistry initialized: collection=%s, "
            "existing_patches=%d, dim=%d",
            collection_name, self._collection.count(), dim,
        )

    def record_attempt(self, task_id: str) -> int:
        """Increment and return the attempt count for a task."""
        self._attempt_counts[task_id] = self._attempt_counts.get(task_id, 0) + 1
        return self._attempt_counts[task_id]

    def register_candidate(
        self,
        task_id: str,
        description: str,
        function_name: str,
        code: str,
        context_vector: NDArray[np.float32],
        passed: bool,
        failure_category: str = "",
    ) -> Optional[PatchEntry]:
        """Register a solution as a patch in ChromaDB.

        Qualification: passed=True AND attempt_count >= min_attempts.
        Returns the PatchEntry if qualified, None otherwise.
        """
        if not passed:
            return None

        attempt_count = self._attempt_counts.get(task_id, 1)
        if attempt_count < self.min_attempts:
            return None

        # Create PatchEntry for compatibility
        patch = PatchEntry(
            task_description=description,
            function_name=function_name,
            code=code,
            context_vector=np.asarray(context_vector, dtype=np.float32).copy(),
            failure_category=failure_category,
            attempt_count=attempt_count,
        )

        # Store in ChromaDB
        try:
            self._collection.add(
                ids=[patch.patch_id],
                embeddings=[context_vector.tolist()],
                metadatas=[{
                    "function_name": function_name,
                    "failure_category": failure_category,
                    "attempt_count": attempt_count,
                    "times_served": 0,
                    "created_at": patch.created_at,
                    "task_description": description[:500],
                    "code": code[:2000],
                }],
                documents=[patch.lesson],
            )
            logger.info(
                "Patch registered in ChromaDB: %s (function=%s, attempts=%d)",
                patch.patch_id, function_name, attempt_count,
            )
        except Exception as e:
            logger.warning("Failed to add patch to ChromaDB: %s", e)
            return None

        return patch

    def search(
        self,
        context_vector: NDArray[np.floating],
        top_k: int = 1,
    ) -> list[PatchEntry]:
        """ANN search via ChromaDB.

        Uses cosine distance for similarity. ChromaDB returns distances
        where distance = 1 - cosine_similarity, so we convert back.

        Returns up to top_k patches sorted by similarity (descending),
        filtered by similarity_threshold.
        """
        if self._collection.count() == 0:
            return []

        query = np.asarray(context_vector, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm < 1e-10:
            return []

        try:
            # Over-fetch to account for threshold filtering
            n_results = min(top_k * 3, self._collection.count())
            results = self._collection.query(
                query_embeddings=[query.tolist()],
                n_results=max(n_results, 1),
                include=["embeddings", "metadatas", "documents", "distances"],
            )
        except Exception as e:
            logger.warning("ChromaDB query failed: %s", e)
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        # Convert results to PatchEntry objects
        patches: list[PatchEntry] = []
        ids = results["ids"][0]
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        embeddings = results["embeddings"][0] if results["embeddings"] else []

        for i, patch_id in enumerate(ids):
            # ChromaDB cosine distance = 1 - cosine_similarity
            distance = distances[i] if i < len(distances) else 1.0
            similarity = 1.0 - distance

            if similarity < self.similarity_threshold:
                continue

            meta = metadatas[i] if i < len(metadatas) else {}
            embedding = embeddings[i] if i < len(embeddings) else [0.0] * self.dim

            patch = PatchEntry(
                patch_id=patch_id,
                task_description=meta.get("task_description", ""),
                function_name=meta.get("function_name", ""),
                code=meta.get("code", ""),
                context_vector=np.array(embedding, dtype=np.float32),
                failure_category=meta.get("failure_category", ""),
                attempt_count=meta.get("attempt_count", 0),
                times_served=meta.get("times_served", 0),
                created_at=meta.get("created_at", 0.0),
            )

            # Increment times_served in ChromaDB
            try:
                new_served = patch.times_served + 1
                self._collection.update(
                    ids=[patch_id],
                    metadatas=[{**meta, "times_served": new_served}],
                )
                patch.times_served = new_served
            except Exception:
                pass  # Non-critical — don't fail the search

            patches.append(patch)
            if len(patches) >= top_k:
                break

        return patches

    @property
    def n_patches(self) -> int:
        """Number of patches currently stored in ChromaDB."""
        try:
            return self._collection.count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Persistence (attempt_counts only — patches live in ChromaDB)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialize attempt_counts and config (ChromaDB is self-persisting)."""
        return {
            "dim": self.dim,
            "min_attempts": self.min_attempts,
            "max_patches": self.max_patches,
            "similarity_threshold": self.similarity_threshold,
            "attempt_counts": dict(self._attempt_counts),
            "backend": "chromadb",
            "n_patches": self.n_patches,
        }

    @classmethod
    def from_state_dict(
        cls,
        d: dict,
        chroma_url: str = "",
        persist_dir: str = "",
        collection_name: str = "mcn_patches",
    ) -> ChromaPatchRegistry:
        """Restore attempt_counts (patches already in ChromaDB)."""
        registry = cls(
            dim=d.get("dim", 18),
            min_attempts=d.get("min_attempts", 2),
            max_patches=d.get("max_patches", 5000),
            similarity_threshold=d.get("similarity_threshold", 0.3),
            chroma_url=chroma_url,
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        registry._attempt_counts = dict(d.get("attempt_counts", {}))
        logger.info(
            "ChromaPatchRegistry restored: %d attempt records, "
            "%d patches in ChromaDB",
            len(registry._attempt_counts), registry.n_patches,
        )
        return registry

    def __repr__(self) -> str:
        return (
            f"ChromaPatchRegistry(n_patches={self.n_patches}, "
            f"tasks_tracked={len(self._attempt_counts)}, dim={self.dim})"
        )
