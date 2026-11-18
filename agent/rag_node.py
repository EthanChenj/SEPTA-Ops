"""
Hybrid RAG retrieval node combining dense pgvector and sparse tsvector search using RRF.
Performs multi-hop retrieval over SEPTA Operating Budget and strategic policy documents.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import math
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False
    np = None

try:
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:  # pragma: no cover
    HAS_SQLALCHEMY = False
    create_engine = None
    text = None

from etl.pdf_ingest import BASELINE_BUDGET_CHUNKS, generate_text_embedding

logger = logging.getLogger("septa_ops.agent.rag_node")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://septa_admin:septa_secure_password@localhost:5432/septa_ops"
)

RRF_K = 60  # Standard Reciprocal Rank Fusion damping constant


@dataclass
class RAGChunk:
    chunk_id: str
    document_title: str
    fiscal_year: str
    page_number: int
    section_header: str
    content: str
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGRetrievalResult:
    query: str
    chunks: List[RAGChunk]
    formatted_context: str
    source_documents: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "chunks": [c.to_dict() for c in self.chunks],
            "formatted_context": self.formatted_context,
            "source_documents": self.source_documents,
        }


class HybridRAGNode:
    """Orchestrates dense cosine similarity and sparse keyword retrieval fused with RRF."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or DATABASE_URL
        self.engine = None
        if HAS_SQLALCHEMY:
            try:
                self.engine = create_engine(self.db_url, connect_args={"connect_timeout": 3})
            except Exception as e:
                logger.debug(f"Hybrid RAG Postgres connection unavailable: {e}")

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Embeds query and retrieves top-k chunks by cosine distance."""
        query_vec = generate_text_embedding(query)

        if self.engine:
            try:
                with self.engine.connect() as conn:
                    stmt = text("""
                        SELECT id, document_title, fiscal_year, page_number, 
                               section_header, content,
                               (1 - (embedding <=> :query_vec)) AS similarity
                        FROM budget_document_chunks
                        ORDER BY embedding <=> :query_vec
                        LIMIT :limit
                    """)
                    rows = conn.execute(stmt, {"query_vec": str(query_vec), "limit": top_k}).fetchall()
                    return [
                        {
                            "id": str(r.id),
                            "document_title": r.document_title,
                            "fiscal_year": r.fiscal_year,
                            "page_number": r.page_number,
                            "section_header": r.section_header,
                            "content": r.content,
                            "score": float(r.similarity),
                        }
                        for r in rows
                    ]
            except Exception as e:
                logger.debug(f"Dense vector DB search skipped: {e}")

        # In-memory vector fallback
        results = []
        if HAS_NUMPY and np is not None:
            q_vec = np.array(query_vec)
            for idx, item in enumerate(BASELINE_BUDGET_CHUNKS):
                doc_emb = np.array(generate_text_embedding(item["content"]))
                sim = float(np.dot(q_vec, doc_emb) / (np.linalg.norm(q_vec) * np.linalg.norm(doc_emb) + 1e-9))
                results.append({
                    "id": f"chunk_baseline_{idx}",
                    "document_title": item["document_title"],
                    "fiscal_year": item["fiscal_year"],
                    "page_number": item["page_number"],
                    "section_header": item["section_header"],
                    "content": item["content"],
                    "score": sim,
                })
        else:
            q_norm = math.sqrt(sum(x * x for x in query_vec)) or 1.0
            for idx, item in enumerate(BASELINE_BUDGET_CHUNKS):
                doc_emb = generate_text_embedding(item["content"])
                dot = sum(a * b for a, b in zip(query_vec, doc_emb))
                d_norm = math.sqrt(sum(x * x for x in doc_emb)) or 1.0
                sim = dot / (q_norm * d_norm)
                results.append({
                    "id": f"chunk_baseline_{idx}",
                    "document_title": item["document_title"],
                    "fiscal_year": item["fiscal_year"],
                    "page_number": item["page_number"],
                    "section_header": item["section_header"],
                    "content": item["content"],
                    "score": sim,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Performs full-text keyword retrieval using tsvector / ts_rank_cd."""
        if self.engine:
            try:
                with self.engine.connect() as conn:
                    stmt = text("""
                        SELECT id, document_title, fiscal_year, page_number,
                               section_header, content,
                               ts_rank_cd(tsv_content, websearch_to_tsquery('english', :query)) AS rank
                        FROM budget_document_chunks
                        WHERE tsv_content @@ websearch_to_tsquery('english', :query)
                        ORDER BY rank DESC
                        LIMIT :limit
                    """)
                    rows = conn.execute(stmt, {"query": query, "limit": top_k}).fetchall()
                    return [
                        {
                            "id": str(r.id),
                            "document_title": r.document_title,
                            "fiscal_year": r.fiscal_year,
                            "page_number": r.page_number,
                            "section_header": r.section_header,
                            "content": r.content,
                            "score": float(r.rank),
                        }
                        for r in rows
                    ]
            except Exception as e:
                logger.debug(f"Sparse text DB search skipped: {e}")

        # In-memory keyword match fallback
        query_words = set(query.lower().split())
        results = []
        for idx, item in enumerate(BASELINE_BUDGET_CHUNKS):
            text_lower = (item["section_header"] + " " + item["content"]).lower()
            matches = sum(1 for w in query_words if w in text_lower)
            if matches > 0:
                results.append({
                    "id": f"chunk_baseline_{idx}",
                    "document_title": item["document_title"],
                    "fiscal_year": item["fiscal_year"],
                    "page_number": item["page_number"],
                    "section_header": item["section_header"],
                    "content": item["content"],
                    "score": float(matches),
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def retrieve(self, query: str, top_k: int = 3) -> RAGRetrievalResult:
        """
        Executes hybrid search and applies Reciprocal Rank Fusion (RRF).
        RRF formula: score = sum(1 / (RRF_K + rank))
        """
        dense_hits = self.dense_search(query, top_k=top_k * 2)
        sparse_hits = self.sparse_search(query, top_k=top_k * 2)

        # Track documents by id
        doc_store: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        # Dense ranking
        for rank, item in enumerate(dense_hits, start=1):
            doc_id = item["id"]
            doc_store[doc_id] = item
            dense_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank))

        # Sparse ranking
        for rank, item in enumerate(sparse_hits, start=1):
            doc_id = item["id"]
            doc_store[doc_id] = item
            sparse_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank))

        # Sort by fused RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]

        fused_chunks: List[RAGChunk] = []
        sources = set()

        for d_id in sorted_doc_ids:
            doc = doc_store[d_id]
            chunk = RAGChunk(
                chunk_id=d_id,
                document_title=doc["document_title"],
                fiscal_year=doc["fiscal_year"],
                page_number=doc["page_number"],
                section_header=doc["section_header"],
                content=doc["content"],
                dense_rank=dense_ranks.get(d_id),
                sparse_rank=sparse_ranks.get(d_id),
                rrf_score=round(rrf_scores[d_id], 5),
            )
            fused_chunks.append(chunk)
            sources.add(f"{chunk.document_title} (p. {chunk.page_number})")

        # Format context string for LLM prompt ingestion
        formatted_context_parts = []
        for i, c in enumerate(fused_chunks, start=1):
            formatted_context_parts.append(
                f"--- [Document {i}: {c.document_title} | Section: {c.section_header} | Page: {c.page_number}] ---\n"
                f"{c.content}\n"
            )
        formatted_context = "\n".join(formatted_context_parts)

        return RAGRetrievalResult(
            query=query,
            chunks=fused_chunks,
            formatted_context=formatted_context,
            source_documents=sorted(sources),
        )


# Singleton instance
rag_node = HybridRAGNode()


if __name__ == "__main__":
    node = HybridRAGNode()
    res = node.retrieve("What is the projected deficit in the FY2026 operating budget?")
    print("Hybrid RAG Result:")
    for chunk in res.chunks:
        print(f"- [{chunk.document_title} p.{chunk.page_number}] RRF Score: {chunk.rrf_score}")
        print(f"  {chunk.content[:120]}...\n")
