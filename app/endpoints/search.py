from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.models.embeddings import model
from app.db.connection import conn
from app.config import DISTANCE_THRESHOLD
from app.utils.prioritization import priority_order
from app.utils.dead import extract_entities_and_deadlines  # <- run extraction on search

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@router.post("/search")
def search_docs(request: QueryRequest, jurisdiction: Optional[str] = "local"):
    query_embedding = model.encode(request.query).tolist()
    query_embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_text, metadata, embedding <-> %s::vector AS distance
            FROM document_chunks
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """, (query_embedding_str, query_embedding_str, request.top_k * 5))
        results = cur.fetchall()

    seen_hashes = set()
    response = []

    for row in results:
        chunk_text = row[0]
        meta = row[1]
        distance = float(row[2])

        # Deduplication
        text_hash = meta.get("text_hash")
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        # Jurisdiction filter
        chunk_jurisdiction = meta.get("jurisdiction", "company")
        if jurisdiction and chunk_jurisdiction != jurisdiction:
            continue

        # Distance threshold filter
        if distance > DISTANCE_THRESHOLD:
            continue

        # --- Run extraction on chunk text ---
        extracted_info = extract_entities_and_deadlines(chunk_text)

        response.append({
            "chunk_text": chunk_text,
            "metadata": {
                "file_name": meta.get("file_name"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "doc_type": meta.get("doc_type"),
                "jurisdiction": chunk_jurisdiction,
                "deadlines": extracted_info.get("deadlines", []),
                "consequences": extracted_info.get("consequences", []),
                "entities": extracted_info.get("entities", []),
                "references": extracted_info.get("references", [])
            },
            "distance": distance,
            "priority": priority_order.get(chunk_jurisdiction, 3)
        })

    # Sort by priority then distance
    response = sorted(response, key=lambda x: (x["priority"], x["distance"]))[:request.top_k]

    return {
        "results": response,
        "threshold": DISTANCE_THRESHOLD
    }
