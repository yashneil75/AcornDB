"""Typed Qdrant store: manage databases, embed texts, search."""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from .config import AcornDBConfig, EmbedConfig, QdrantConfig
from .embedder import EmbeddingGemmaEmbedder
from .models import DatabaseInfo, Distance, Document, PointId, SearchHit

_TEXT_KEY = "text"


def _to_distance(distance: Distance):  # type: ignore[no-untyped-def]
    from qdrant_client import models

    mapping = {
        "Cosine": models.Distance.COSINE,
        "Euclid": models.Distance.EUCLID,
        "Dot": models.Distance.DOT,
    }
    return mapping[distance]


class AcornDB:
    """One typed entry point for everything.

    Example:
        from acorndb import AcornDB, QdrantConfig

        db = AcornDB(qdrant=QdrantConfig(url="http://localhost:6333"))
        db.create_database("notes")
        db.add("notes", ["hello world", "qdrant is fast"])
        hits = db.search("notes", "hello")
        print(hits[0].text, hits[0].score)
    """

    def __init__(
        self,
        config: AcornDBConfig | None = None,
        *,
        qdrant: QdrantConfig | None = None,
        embed: EmbedConfig | None = None,
        embedder: EmbeddingGemmaEmbedder | None = None,
    ) -> None:
        if config is not None and (qdrant is not None or embed is not None):
            raise ValueError("pass either config= or qdrant=/embed=, not both")
        self.config = config or AcornDBConfig(
            qdrant=qdrant or QdrantConfig(),
            embed=embed or EmbedConfig(),
        )
        self._embedder = embedder or EmbeddingGemmaEmbedder(self.config.embed)
        self._client = None  # QdrantClient, created lazily

    # -- embedder passthrough ("a thing to embed something") -------------
    @property
    def embedder(self) -> EmbeddingGemmaEmbedder:
        return self._embedder

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        return self._embedder.embed(text)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many strings."""
        return self._embedder.embed_many(texts)

    # -- client ----------------------------------------------------------
    def _get_client(self):  # type: ignore[no-untyped-def]
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required. Install with `pip install acorndb[qdrant]`."
            ) from exc
        q = self.config.qdrant
        if q.url:
            self._client = QdrantClient(
                url=q.url, api_key=q.api_key, prefer_grpc=q.prefer_grpc, timeout=q.timeout
            )
        elif q.path:
            self._client = QdrantClient(path=q.path)
        elif q.in_memory:
            self._client = QdrantClient(":memory:")
        else:
            # Default to in-memory so `AcornDB()` works with zero infra.
            self._client = QdrantClient(":memory:")
        return self._client

    # -- databases ("add delete or list db's") ---------------------------
    def create_database(
        self,
        name: str,
        *,
        vector_size: int | None = None,
        distance: Distance | None = None,
        overwrite: bool = False,
    ) -> DatabaseInfo:
        """Create a collection. `overwrite=True` deletes it first if it exists."""
        from qdrant_client import models

        if not name:
            raise ValueError("name must be a non-empty string")
        client = self._get_client()
        if client.collection_exists(name):
            if not overwrite:
                raise ValueError(f"database {name!r} already exists")
            client.delete_collection(name)
        size = vector_size or self.dimension
        dist = distance or self.config.default_distance
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=_to_distance(dist)),
        )
        return DatabaseInfo(name=name, vector_size=size, distance=dist)

    def delete_database(self, name: str) -> bool:
        """Delete a collection. Returns True if it existed."""
        client = self._get_client()
        if not client.collection_exists(name):
            return False
        client.delete_collection(name)
        return True

    def list_databases(self) -> list[DatabaseInfo]:
        """List all collections with their vector params + point counts."""
        client = self._get_client()
        out: list[DatabaseInfo] = []
        for col in client.get_collections().collections:
            info = client.get_collection(col.name)
            params = info.config.params.vectors  # type: ignore[union-attr]
            # Single-vector collections expose .size/.distance directly.
            size = getattr(params, "size", self.dimension)
            dist = str(getattr(params, "distance", "?")).split(".")[-1].lower().capitalize()
            out.append(
                DatabaseInfo(
                    name=col.name,
                    vector_size=size,
                    distance=dist,
                    points_count=info.points_count,
                )
            )
        return out

    def database_exists(self, name: str) -> bool:
        return self._get_client().collection_exists(name)

    # -- write -----------------------------------------------------------
    def add(
        self,
        database: str,
        texts: Sequence[str] | Sequence[Document],
        *,
        payloads: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[PointId] | None = None,
    ) -> list[PointId]:
        """Embed `texts` and upsert them into `database`.

        Accepts either `list[str]` or `list[Document]`.
        Returns the point ids used.
        """
        from qdrant_client import models

        if not self.database_exists(database):
            # Auto-create with the embedder's dimension so first use "just works".
            self.create_database(database)
        docs = self._coerce_documents(texts, payloads=payloads, ids=ids)
        vectors = self._embedder.embed_many([d.text for d in docs])
        points = [
            models.PointStruct(
                id=d.id if d.id is not None else str(uuid.uuid4()),
                vector=vec,
                payload={_TEXT_KEY: d.text, **d.payload},
            )
            for d, vec in zip(docs, vectors)
        ]
        self._get_client().upsert(collection_name=database, points=points)
        return [p.id for p in points]  # type: ignore[return-value]

    def add_vectors(
        self,
        database: str,
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str] | None = None,
        *,
        payloads: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[PointId] | None = None,
    ) -> list[PointId]:
        """Upsert pre-computed vectors (skips the embedder)."""
        from qdrant_client import models

        if not vectors:
            return []
        if not self.database_exists(database):
            dim = len(vectors[0])
            self.create_database(database, vector_size=dim)
        n = len(vectors)
        texts = list(texts) if texts is not None else ["" for _ in range(n)]
        if len(texts) != n:
            raise ValueError("texts and vectors must have the same length")
        payloads = list(payloads) if payloads is not None else [{} for _ in range(n)]
        ids = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in range(n)]
        if not (len(payloads) == len(ids) == n):
            raise ValueError("payloads/ids must match vectors length")
        points = [
            models.PointStruct(
                id=pid,
                vector=list(vec),
                payload=({} if not t else {_TEXT_KEY: t}) | dict(pl),
            )
            for pid, vec, t, pl in zip(ids, vectors, texts, payloads)
        ]
        self._get_client().upsert(collection_name=database, points=points)
        return list(ids)

    def delete_points(self, database: str, ids: Sequence[PointId]) -> None:
        from qdrant_client import models

        self._get_client().delete(
            collection_name=database,
            points_selector=models.PointIdsList(points=list(ids)),
        )

    # -- search ("another to search") ------------------------------------
    def search(
        self,
        database: str,
        query: str | Sequence[float],
        *,
        limit: int = 10,
        filter: Any | None = None,  # qdrant_client.models.Filter
        with_payload: bool = True,
    ) -> list[SearchHit]:
        """Semantic search by text (auto-embedded) or raw vector."""
        client = self._get_client()
        vec = self._embedder.embed(query) if isinstance(query, str) else list(query)
        res = client.query_points(
            collection_name=database,
            query=vec,
            query_filter=filter,
            limit=limit,
            with_payload=with_payload,
        )
        hits: list[SearchHit] = []
        for p in res.points:
            payload = dict(p.payload or {}) if with_payload else {}
            text = payload.pop(_TEXT_KEY, None)
            if isinstance(text, str) is False:
                text = None if text is None else str(text)
            hits.append(
                SearchHit(id=p.id, score=float(p.score), text=text, payload=payload)
            )
        return hits

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _coerce_documents(
        texts: Sequence[str] | Sequence[Document],
        *,
        payloads: Sequence[dict[str, Any]] | None,
        ids: Sequence[PointId] | None,
    ) -> list[Document]:
        if not texts:
            raise ValueError("texts must be non-empty")
        if isinstance(texts[0], Document):
            docs = list(texts)  # type: ignore[arg-type]
            if payloads is not None or ids is not None:
                raise ValueError("pass payloads/ids inside Document, not separately")
            return docs  # type: ignore[return-value]
        strs = list(texts)  # type: ignore[arg-type]
        n = len(strs)
        payloads = list(payloads) if payloads is not None else [{} for _ in range(n)]
        ids = list(ids) if ids is not None else [None for _ in range(n)]  # type: ignore[list-item]
        if len(payloads) != n or len(ids) != n:
            raise ValueError("payloads/ids must match texts length")
        return [
            Document(text=t, payload=dict(pl), id=pid)
            for t, pl, pid in zip(strs, payloads, ids)
        ]
