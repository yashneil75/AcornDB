"""Typed Qdrant store: manage databases, embed texts, search."""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from .config import AcornDBConfig, EmbedConfig, QdrantConfig
from .embedder import EmbeddingGemmaEmbedder
from .models import (
    AudioDocument,
    DatabaseInfo,
    Distance,
    Document,
    ImageDocument,
    Modality,
    PointId,
    SearchHit,
    VideoDocument,
)

_TEXT_KEY = "text"
_IMAGE_KEY = "image"
_VIDEO_KEY = "video"
_AUDIO_KEY = "audio"
_MODALITY_KEY = "modality"


def _infer_modality(mod: Any, text: Any, img: Any, vid: Any, aud: Any) -> Modality:
    if mod in ("text", "image", "video", "audio"):
        return mod  # type: ignore[return-value]
    # Legacy points (written before modality was stored): guess from payload.
    if img is not None:
        return "image"
    if vid is not None:
        return "video"
    if aud is not None:
        return "audio"
    return "text"


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
        embedder: Any | None = None,
    ) -> None:
        """`embedder` is any object with `dimension` + `embed`/`embed_many`.

        Pass `EmbeddingGemmaEmbedder` (GGUF text, vision via lazy
        sentence-transformers) or `EmbeddingGemmaMultimodalEmbedder`
        (full-precision text + image + video, best for cross-modal search).
        """
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
    def embedder(self) -> Any:
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

    def _media_method(self, name: str):  # type: ignore[no-untyped-def]
        fn = getattr(self._embedder, name, None)
        if fn is None:
            raise RuntimeError(
                f"the configured embedder ({type(self._embedder).__name__}) "
                f"does not support {name}(). Use EmbeddingGemmaEmbedder or "
                "EmbeddingGemmaMultimodalEmbedder (needs "
                "`pip install acorndb[multimodal]` for image/video/audio and "
                "composed inputs)."
            )
        return fn

    def embed_image(
        self, image: Any, *, prompt_name: str | None = None
    ) -> list[float]:
        """Embed one image (path/URL, bytes, PIL, numpy) -> unit-norm vector."""
        return self._media_method("embed_image")(image, prompt_name=prompt_name)

    def embed_images(
        self,
        images: Sequence[Any],
        batch_size: int = 32,
        *,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Embed many images."""
        return self._media_method("embed_images")(
            images, batch_size=batch_size, prompt_name=prompt_name
        )

    def embed_video(
        self, video: Any, *, prompt_name: str | None = None
    ) -> list[float]:
        """Embed one video (path/URL, bytes, frame list) -> unit-norm vector."""
        return self._media_method("embed_video")(video, prompt_name=prompt_name)

    def embed_videos(
        self,
        videos: Sequence[Any],
        batch_size: int = 32,
        *,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Embed many videos."""
        return self._media_method("embed_videos")(
            videos, batch_size=batch_size, prompt_name=prompt_name
        )

    def embed_audio(
        self, audio: Any, *, prompt_name: str | None = None
    ) -> list[float]:
        """Embed audio (path/URL, bytes, waveform, (wave, sr)) -> vector."""
        return self._media_method("embed_audio")(audio, prompt_name=prompt_name)

    def embed_audios(
        self,
        audios: Sequence[Any],
        batch_size: int = 32,
        *,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Embed many audio inputs."""
        return self._media_method("embed_audios")(
            audios, batch_size=batch_size, prompt_name=prompt_name
        )

    def embed_composed(
        self, item: dict[str, Any], *, prompt_name: str | None = None
    ) -> list[float]:
        """Embed one text+media dict in a single pass -> unit-norm vector."""
        return self._media_method("embed_composed")(item, prompt_name=prompt_name)

    def embed_composed_many(
        self,
        items: Sequence[dict[str, Any]],
        batch_size: int = 32,
        *,
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Embed many text+media dicts."""
        return self._media_method("embed_composed_many")(
            items, batch_size=batch_size, prompt_name=prompt_name
        )

    def embed_interleaved(
        self,
        text: str,
        images: Any,
        *,
        prompt_name: str | None = None,
    ) -> list[float]:
        """Single-pass joint embedding of text + image(s)."""
        return self._media_method("embed_interleaved")(
            text, images, prompt_name=prompt_name
        )

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

    def add_images(
        self,
        database: str,
        images: Sequence[Any] | Sequence[ImageDocument],
        *,
        refs: Sequence[str | None] | None = None,
        payloads: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[PointId] | None = None,
    ) -> list[PointId]:
        """Embed `images` and upsert them into `database`.

        Accepts either raw image inputs (path/URL, bytes, PIL, numpy) or
        `list[ImageDocument]`. Each point payload carries
        `{"modality": "image", "image": <ref>, ...}`. Returns point ids.
        """
        from .multimodal import media_ref

        if not self.database_exists(database):
            self.create_database(database)
        docs = self._coerce_image_documents(images, refs=refs, payloads=payloads, ids=ids)
        vectors = self.embed_images([d.image for d in docs])
        points = self._media_points(docs, vectors, media_ref, "image")
        self._get_client().upsert(collection_name=database, points=points)
        return [p.id for p in points]  # type: ignore[return-value]

    def add_videos(
        self,
        database: str,
        videos: Sequence[Any] | Sequence[VideoDocument],
        *,
        refs: Sequence[str | None] | None = None,
        payloads: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[PointId] | None = None,
    ) -> list[PointId]:
        """Embed `videos` and upsert them into `database`.

        Accepts either raw video inputs (path/URL, bytes, frame lists) or
        `list[VideoDocument]`. Each point payload carries
        `{"modality": "video", "video": <ref>, ...}`. Returns point ids.
        """
        from .multimodal import media_ref

        if not self.database_exists(database):
            self.create_database(database)
        docs = self._coerce_video_documents(videos, refs=refs, payloads=payloads, ids=ids)
        vectors = self.embed_videos([d.video for d in docs])
        points = self._media_points(docs, vectors, media_ref, "video")
        self._get_client().upsert(collection_name=database, points=points)
        return [p.id for p in points]  # type: ignore[return-value]

    def add_audios(
        self,
        database: str,
        audios: Sequence[Any] | Sequence[AudioDocument],
        *,
        refs: Sequence[str | None] | None = None,
        payloads: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[PointId] | None = None,
    ) -> list[PointId]:
        """Embed `audios` and upsert them into `database`.

        Accepts either raw audio inputs (path/URL, bytes, waveforms,
        `(wave, sr)` tuples) or `list[AudioDocument]`. Each point payload
        carries `{"modality": "audio", "audio": <ref>, ...}`. Returns ids.
        """
        from .multimodal import media_ref

        if not self.database_exists(database):
            self.create_database(database)
        docs = self._coerce_audio_documents(audios, refs=refs, payloads=payloads, ids=ids)
        vectors = self.embed_audios([d.audio for d in docs])
        points = self._media_points(docs, vectors, media_ref, "audio")
        self._get_client().upsert(collection_name=database, points=points)
        return [p.id for p in points]  # type: ignore[return-value]

    @staticmethod
    def _media_points(  # type: ignore[no-untyped-def]
        docs: Sequence[Any],
        vectors: Sequence[Sequence[float]],
        media_ref: Any,
        kind: str,
    ):
        from qdrant_client import models

        key = {"image": _IMAGE_KEY, "video": _VIDEO_KEY, "audio": _AUDIO_KEY}[kind]
        return [
            models.PointStruct(
                id=d.id if d.id is not None else str(uuid.uuid4()),
                vector=list(vec),
                payload={
                    _MODALITY_KEY: kind,
                    key: media_ref(getattr(d, kind), kind, d.ref),
                    **d.payload,
                },
            )
            for d, vec in zip(docs, vectors)
        ]

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
        query: str | Sequence[float] | None = None,
        *,
        image: Any | None = None,
        video: Any | None = None,
        audio: Any | None = None,
        composed: dict[str, Any] | None = None,
        prompt_name: str | None = None,
        limit: int = 10,
        filter: Any | None = None,  # qdrant_client.models.Filter
        with_payload: bool = True,
    ) -> list[SearchHit]:
        """Semantic search by text (auto-embedded), image, video, audio,
        composed text+media dict, or raw vector.

        Pass exactly one of `query` / `image` / `video` / `audio` /
        `composed`. All modalities share EmbeddingGemma-2's unified space, so
        a text query finds images, clips and recordings, and vice versa.
        `prompt_name` (e.g. `"query"`) applies to the encoded query.
        """
        given = sum(
            [
                query is not None,
                image is not None,
                video is not None,
                audio is not None,
                composed is not None,
            ]
        )
        if given != 1:
            raise ValueError(
                "pass exactly one of query / image / video / audio / composed"
            )
        client = self._get_client()
        if image is not None:
            vec = self.embed_image(image, prompt_name=prompt_name)
        elif video is not None:
            vec = self.embed_video(video, prompt_name=prompt_name)
        elif audio is not None:
            vec = self.embed_audio(audio, prompt_name=prompt_name)
        elif composed is not None:
            vec = self.embed_composed(composed, prompt_name=prompt_name)
        elif isinstance(query, str):
            vec = self._embedder.embed(query)
        else:
            vec = list(query)  # type: ignore[arg-type]
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
            img = payload.pop(_IMAGE_KEY, None)
            img = None if img is None else img if isinstance(img, str) else str(img)
            vid = payload.pop(_VIDEO_KEY, None)
            vid = None if vid is None else vid if isinstance(vid, str) else str(vid)
            aud = payload.pop(_AUDIO_KEY, None)
            aud = None if aud is None else aud if isinstance(aud, str) else str(aud)
            mod = payload.pop(_MODALITY_KEY, None)
            modality: Modality = _infer_modality(mod, text, img, vid, aud)
            hits.append(
                SearchHit(
                    id=p.id,
                    score=float(p.score),
                    text=text,
                    payload=payload,
                    modality=modality,
                    image=img,
                    video=vid,
                    audio=aud,
                )
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

    @staticmethod
    def _coerce_image_documents(
        images: Sequence[Any] | Sequence[ImageDocument],
        *,
        refs: Sequence[str | None] | None,
        payloads: Sequence[dict[str, Any]] | None,
        ids: Sequence[PointId] | None,
    ) -> list[ImageDocument]:
        if not images:
            raise ValueError("images must be non-empty")
        if isinstance(images[0], ImageDocument):
            docs = list(images)  # type: ignore[arg-type]
            if refs is not None or payloads is not None or ids is not None:
                raise ValueError("pass refs/payloads/ids inside ImageDocument, not separately")
            return docs  # type: ignore[return-value]
        raws = list(images)  # type: ignore[arg-type]
        n = len(raws)
        refs = list(refs) if refs is not None else [None for _ in range(n)]
        payloads = list(payloads) if payloads is not None else [{} for _ in range(n)]
        ids = list(ids) if ids is not None else [None for _ in range(n)]  # type: ignore[list-item]
        if not (len(refs) == len(payloads) == len(ids) == n):
            raise ValueError("refs/payloads/ids must match images length")
        return [
            ImageDocument(image=im, payload=dict(pl), id=pid, ref=rf)
            for im, pl, pid, rf in zip(raws, payloads, ids, refs)
        ]

    @staticmethod
    def _coerce_video_documents(
        videos: Sequence[Any] | Sequence[VideoDocument],
        *,
        refs: Sequence[str | None] | None,
        payloads: Sequence[dict[str, Any]] | None,
        ids: Sequence[PointId] | None,
    ) -> list[VideoDocument]:
        if not videos:
            raise ValueError("videos must be non-empty")
        if isinstance(videos[0], VideoDocument):
            docs = list(videos)  # type: ignore[arg-type]
            if refs is not None or payloads is not None or ids is not None:
                raise ValueError("pass refs/payloads/ids inside VideoDocument, not separately")
            return docs  # type: ignore[return-value]
        raws = list(videos)  # type: ignore[arg-type]
        n = len(raws)
        refs = list(refs) if refs is not None else [None for _ in range(n)]
        payloads = list(payloads) if payloads is not None else [{} for _ in range(n)]
        ids = list(ids) if ids is not None else [None for _ in range(n)]  # type: ignore[list-item]
        if not (len(refs) == len(payloads) == len(ids) == n):
            raise ValueError("refs/payloads/ids must match videos length")
        return [
            VideoDocument(video=v, payload=dict(pl), id=pid, ref=rf)
            for v, pl, pid, rf in zip(raws, payloads, ids, refs)
        ]

    @staticmethod
    def _coerce_audio_documents(
        audios: Sequence[Any] | Sequence[AudioDocument],
        *,
        refs: Sequence[str | None] | None,
        payloads: Sequence[dict[str, Any]] | None,
        ids: Sequence[PointId] | None,
    ) -> list[AudioDocument]:
        if not audios:
            raise ValueError("audios must be non-empty")
        if isinstance(audios[0], AudioDocument):
            docs = list(audios)  # type: ignore[arg-type]
            if refs is not None or payloads is not None or ids is not None:
                raise ValueError("pass refs/payloads/ids inside AudioDocument, not separately")
            return docs  # type: ignore[return-value]
        raws = list(audios)  # type: ignore[arg-type]
        n = len(raws)
        refs = list(refs) if refs is not None else [None for _ in range(n)]
        payloads = list(payloads) if payloads is not None else [{} for _ in range(n)]
        ids = list(ids) if ids is not None else [None for _ in range(n)]  # type: ignore[list-item]
        if not (len(refs) == len(payloads) == len(ids) == n):
            raise ValueError("refs/payloads/ids must match audios length")
        return [
            AudioDocument(audio=a, payload=dict(pl), id=pid, ref=rf)
            for a, pl, pid, rf in zip(raws, payloads, ids, refs)
        ]
