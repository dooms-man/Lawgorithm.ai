from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.models.embeddings import model
from app.models.llm_client import hf_client, LLM_MODEL
from app.db.connection import conn
from app.config import DISTANCE_THRESHOLD
from app.utils.prioritization import priority_order
from app.utils.dead import extract_entities_and_deadlines  # <- import extraction
from app.utils.semantic_matching  import find_top_regulations_by_embedding
from app.db.queries import get_contract_chunks,store_clause_regulation_mapping, get_all_regulation_chunks
from app.models.llm_client import evaluate_clause_with_llm , query_llm
import time
from app.config import TOP_K_REGULATIONS, DISTANCE_THRESHOLD
from sentence_transformers import SentenceTransformer
import numpy as np





router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@router.post("/rag")
def rag_response(request: QueryRequest, jurisdiction: Optional[str] = "local"):
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
    filtered = []

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

        if distance <= DISTANCE_THRESHOLD:
            # --- Run extraction here ---
            extracted_info = extract_entities_and_deadlines(chunk_text)

            filtered.append({
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
    filtered = sorted(filtered, key=lambda x: (x["priority"], x["distance"]))[:request.top_k]

    # Build context with metadata
    if filtered:
        context = "\n\n".join(
            f"Chunk {i+1}:\nText: {row['chunk_text']}\nEntities: {row['metadata']['entities']}\n"
            f"Deadlines: {row['metadata']['deadlines']}\nConsequences: {row['metadata']['consequences']}\n"
            f"References: {row['metadata']['references']}"
            for i, row in enumerate(filtered)
        )
    else:
        context = "No context found."

    messages = [
        {"role": "system", "content": "You are a legal AI that explains compliance, deadlines, and penalties clearly."},
        {"role": "user", "content": f"Question: {request.query}\n\nContext:\n{context}\n\nAnswer:"}
    ]

    response = hf_client.chat_completion(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    answer = response.choices[0].message["content"]

    return {
        "query": request.query,
        "context_chunks": filtered,
        "response": answer,
        "threshold": DISTANCE_THRESHOLD
    }

    # --- Step 6: Build messages for LLM ---
    messages = [
        {"role": "system", "content": "You are a legal AI that explains compliance, deadlines, and penalties clearly."},
        {"role": "user", "content": f"Question: {request.query}\n\nContext:\n{context}\n\nAnswer:"}
    ]

    # --- Step 7: Call LLM ---
    response = hf_client.chat_completion(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    answer = response.choices[0].message["content"]

    # --- Step 8: Return structured response ---
    return {
        "query": request.query,
        "context_chunks": filtered,
        "response": answer,
        "threshold": DISTANCE_THRESHOLD
    }

