from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from app.errors import api_error
from app.integrations.embedding_http_client import EmbeddingHTTPClient
from app.integrations.qdrant_client import QdrantClient
from app.repositories.sqlite_repository import SQLiteRepository, decode_collection, utc_now
from app.schemas import DocumentInput, DocumentsUpsertRequest, JobStatus
from app.services.chunk_service import ChunkService
from app.utils.hashing import qdrant_point_id, sha256_json, sha256_text


class DocumentService:
    def __init__(
        self,
        repo: SQLiteRepository,
        qdrant: QdrantClient,
        embedding: EmbeddingHTTPClient,
        chunker: ChunkService,
    ) -> None:
        self.repo = repo
        self.qdrant = qdrant
        self.embedding = embedding
        self.chunker = chunker

    def upsert(self, request: DocumentsUpsertRequest) -> dict[str, Any]:
        self._collection(request.collection)
        if request.indexing.mode not in {"sync", "async"}:
            raise api_error(400, "INVALID_REQUEST", "indexing.mode must be sync or async")
        if request.indexing.mode == "async":
            jobs = [
                self._create_job(request.collection, "upsert", doc.id, doc)
                for doc in request.documents
            ]
            return {
                "collection": request.collection,
                "accepted": len(request.documents),
                "upserted": 0,
                "skipped": 0,
                "jobs": jobs,
            }
        upserted = 0
        skipped = 0
        jobs = []
        for document in request.documents:
            job = self._create_job(
                request.collection,
                "upsert",
                document.id,
                document,
                "processing",
            )
            try:
                result = self._index_document(request.collection, document)
                upserted += 0 if result["skipped"] else 1
                skipped += 1 if result["skipped"] else 0
                self.repo.update_job(
                    job["job_id"],
                    status=JobStatus.DONE.value,
                    finished_at=utc_now(),
                    last_error=None,
                )
                jobs.append({**job, "status": JobStatus.DONE.value})
            except Exception as exc:
                self.repo.update_job(
                    job["job_id"],
                    status=JobStatus.FAILED.value,
                    finished_at=utc_now(),
                    last_error=str(exc),
                )
                raise
        return {
            "collection": request.collection,
            "accepted": len(request.documents),
            "upserted": upserted,
            "skipped": skipped,
            "jobs": jobs,
        }

    def get(self, collection: str, document_id: str) -> dict[str, Any]:
        document = self.repo.get_document(collection, document_id)
        if not document:
            raise api_error(404, "DOCUMENT_NOT_FOUND", f"Document not found: {document_id}")
        chunks = self.repo.get_chunks(collection, document_id)
        return {
            "collection": collection,
            "document_id": document_id,
            "source": document["source"],
            "doc_type": document["doc_type"],
            "title": document["title"],
            "status": document["status"],
            "text_hash": document["text_hash"],
            "metadata": json.loads(document["metadata_json"]),
            "indexed_at": document["indexed_at"],
            "chunks": [
                {
                    "chunk_id": chunk["id"],
                    "chunk_index": chunk["chunk_index"],
                    "text_hash": chunk["text_hash"],
                    "token_count": chunk["token_count"],
                }
                for chunk in chunks
            ],
        }

    def delete(self, collection: str, document_id: str) -> dict[str, Any]:
        deleted_chunks = self.repo.delete_document(collection, document_id)
        self.qdrant.delete_document_points(collection, document_id)
        return {
            "collection": collection,
            "document_id": document_id,
            "deleted_chunks": deleted_chunks,
            "deleted": True,
        }

    def archive(self, collection: str, document_id: str) -> dict[str, Any]:
        return self._set_status(collection, document_id, "archived")

    def restore(
        self,
        collection: str,
        document_id: str,
        status: str = "published",
    ) -> dict[str, Any]:
        return self._set_status(collection, document_id, status)

    def reindex(
        self,
        collection: str,
        documents: list[DocumentInput],
        delete_missing: bool,
    ) -> dict[str, Any]:
        self._collection(collection)
        seen = {document.id for document in documents}
        for document in documents:
            self._index_document(collection, document, force=True)
        if delete_missing:
            for row in self.repo.list_documents(collection):
                if row["id"] not in seen:
                    self.delete(collection, row["id"])
        job = self._create_job(
            collection,
            "reindex",
            None,
            {"documents": [doc.model_dump(mode="json") for doc in documents]},
            status=JobStatus.DONE.value,
        )
        return {"collection": collection, "accepted": len(documents), "job_id": job["job_id"]}

    def rebuild_collection(
        self,
        collection: str,
        documents: list[DocumentInput],
        mode: str,
    ) -> dict[str, Any]:
        if mode not in {"replace", "merge"}:
            raise api_error(400, "INVALID_REQUEST", "mode must be replace or merge")
        self._collection(collection)
        if mode == "replace":
            for row in self.repo.list_documents(collection):
                self.delete(collection, row["id"])
        for document in documents:
            self._index_document(collection, document, force=True)
        job = self._create_job(
            collection,
            "rebuild",
            None,
            {"mode": mode, "documents": [doc.model_dump(mode="json") for doc in documents]},
            status=JobStatus.DONE.value,
        )
        return {"collection": collection, "accepted": len(documents), "job_id": job["job_id"]}

    def list_jobs(
        self,
        collection: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return [self._decode_job(row) for row in self.repo.list_jobs(collection, status, limit)]

    def get_job(self, job_id: str) -> dict[str, Any]:
        row = self.repo.get_job(job_id)
        if not row:
            raise api_error(404, "INDEX_JOB_FAILED", f"Job not found: {job_id}")
        return self._decode_job(row)

    def retry_job(self, job_id: str) -> dict[str, Any]:
        self.process_job(job_id)
        return self.get_job(job_id)

    def process_job(self, job_id: str) -> None:
        row = self.repo.get_job(job_id)
        if not row:
            raise api_error(404, "INDEX_JOB_FAILED", f"Job not found: {job_id}")
        payload = json.loads(row["payload_json"])
        self.repo.update_job(
            job_id,
            status=JobStatus.PROCESSING.value,
            retry_count=int(row["retry_count"]) + 1,
            started_at=utc_now(),
        )
        try:
            if row["action"] == "upsert":
                self._index_document(
                    row["collection"],
                    DocumentInput.model_validate(payload),
                    force=True,
                )
            self.repo.update_job(
                job_id,
                status=JobStatus.DONE.value,
                finished_at=utc_now(),
                last_error=None,
            )
        except Exception as exc:
            self.repo.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=utc_now(),
                last_error=str(exc),
            )

    def delete_job(self, job_id: str) -> dict[str, Any]:
        deleted = self.repo.delete_job(job_id)
        if not deleted:
            raise api_error(404, "INDEX_JOB_FAILED", f"Job not found: {job_id}")
        return {"job_id": job_id, "deleted": True}

    def _index_document(
        self,
        collection: str,
        document: DocumentInput,
        force: bool = False,
    ) -> dict[str, Any]:
        collection_row = self._collection(collection)
        metadata = {**document.metadata}
        metadata.setdefault("status", "published")
        metadata.update(
            {
                "source": document.source,
                "doc_type": document.doc_type,
                "document_id": document.id,
            }
        )
        text_hash = sha256_text(document.text)
        metadata_hash = sha256_json(metadata)
        existing = self.repo.get_document(collection, document.id)
        if (
            not force
            and existing
            and existing["text_hash"] == text_hash
            and sha256_text(existing["metadata_json"]) == metadata_hash
        ):
            return {"skipped": True}
        chunks = self.chunker.chunk_document(
            collection=collection,
            document_id=document.id,
            title=document.title,
            text=document.text,
            metadata=metadata,
            strategy=collection_row["chunk_strategy"],
        )
        model, dimension, vectors = self.embedding.embed([chunk.text for chunk in chunks])
        if dimension != int(collection_row["embedding_dimension"]):
            raise api_error(
                400,
                "DIMENSION_MISMATCH",
                f"Expected {collection_row['embedding_dimension']}, got {dimension}",
                {"model": model},
            )
        chunk_dicts = [
            {
                "id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "text_hash": chunk.text_hash,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        self.repo.upsert_document(
            {
                "id": document.id,
                "collection": collection,
                "source": document.source,
                "doc_type": document.doc_type,
                "title": document.title,
                "text_hash": text_hash,
                "metadata": metadata,
                "status": metadata["status"],
                "updated_at": (document.updated_at or datetime.now().astimezone()).isoformat(),
            }
        )
        self.repo.replace_chunks(collection, document.id, chunk_dicts)
        points = [
            {
                "id": qdrant_point_id(collection, chunk.id),
                "vector": vector,
                "payload": {
                    **chunk.metadata,
                    "collection": collection,
                    "document_id": document.id,
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "title": document.title,
                },
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.qdrant.delete_document_points(collection, document.id)
        self.qdrant.upsert_points(collection, points)
        return {"skipped": False}

    def _set_status(self, collection: str, document_id: str, status: str) -> dict[str, Any]:
        if not self.repo.get_document(collection, document_id):
            raise api_error(404, "DOCUMENT_NOT_FOUND", f"Document not found: {document_id}")
        self.repo.update_document_status(collection, document_id, status)
        self.qdrant.set_document_payload(collection, document_id, {"status": status})
        return {"collection": collection, "document_id": document_id, "status": status}

    def _collection(self, collection: str) -> dict[str, Any]:
        row = self.repo.get_collection(collection)
        if not row:
            raise api_error(404, "COLLECTION_NOT_FOUND", f"Collection not found: {collection}")
        return decode_collection(row)

    def _create_job(
        self,
        collection: str,
        action: str,
        document_id: str | None,
        payload: Any,
        status: str = JobStatus.PENDING.value,
    ) -> dict[str, Any]:
        job_id = f"job_{uuid.uuid4().hex}"
        payload_json = (
            payload.model_dump(mode="json")
            if hasattr(payload, "model_dump")
            else payload
        )
        job = {
            "job_id": job_id,
            "collection": collection,
            "action": action,
            "document_id": document_id,
            "payload": payload_json,
            "status": status,
            "started_at": utc_now() if status == JobStatus.PROCESSING.value else None,
            "finished_at": utc_now() if status == JobStatus.DONE.value else None,
        }
        self.repo.insert_job(job)
        return {"job_id": job_id, "document_id": document_id, "status": status}

    def _decode_job(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "collection": row["collection"],
            "action": row["action"],
            "document_id": row["document_id"],
            "payload": json.loads(row["payload_json"]),
            "status": row["status"],
            "retry_count": row["retry_count"],
            "max_retries": row["max_retries"],
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }
