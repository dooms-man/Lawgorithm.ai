# app/db/queries.py
from app.db.connection import conn
import json
import hashlib
from datetime import datetime
from app.models.embeddings import model2 as embedding_model

# -------------------------
# Internal Compliance Chunks
# -------------------------
def insert_chunk(text, embedding, metadata):
    """
    Inserts a chunk into the 'document_chunks' table if it doesn't already exist.
    Used for internal compliance documents.
    """
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    metadata["text_hash"] = text_hash
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM document_chunks
            WHERE (metadata->>'file_name') = %s
              AND (metadata->>'chunk_index') = %s
              AND (metadata->>'text_hash') = %s
            LIMIT 1;
        """, (metadata["file_name"], str(metadata["chunk_index"]), text_hash))
        if cur.fetchone():
            return  # Already exists

        cur.execute("""
            INSERT INTO document_chunks (chunk_text, embedding, metadata)
            VALUES (%s, %s::vector, %s)
        """, (text, embedding_str, json.dumps(metadata)))


# -------------------------
# Regulation Chunks
# -------------------------
def insert_regulation_chunk(text, embedding, metadata):
    """
    Inserts a chunk into the 'regulations' table if it doesn't already exist.
    Used specifically for regulations to allow separate gap detection.
    """
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    metadata["text_hash"] = text_hash
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM regulations
            WHERE (metadata->>'file_name') = %s
              AND (metadata->>'chunk_index') = %s
              AND (metadata->>'text_hash') = %s
            LIMIT 1;
        """, (metadata["file_name"], str(metadata["chunk_index"]), text_hash))
        if cur.fetchone():
            return  # Already exists

        cur.execute("""
            INSERT INTO regulations (chunk_text, embedding, metadata)
            VALUES (%s, %s::vector, %s)
        """, (text, embedding_str, json.dumps(metadata)))


# -------------------------
# Compliance Flags
# -------------------------
def store_compliance_flag(suggestion: dict, severity="high", assigned_to=None):
    """
    Stores a compliance gap flag in the 'compliance_flags' table with optional assignment.
    """
    hash_input = (
        suggestion.get("clause", "") +
        suggestion.get("evidence_sentences", "") +
        json.dumps(suggestion.get("action_steps", []))
    )
    flag_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO compliance_flags 
            (regulation_id, clause, evidence_sentences, confidence, action_steps, explanation_evidence, severity, status, assigned_to, page_reference, doc_reference, hash)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id;
        """, (
            suggestion.get("regulation_id"),
            suggestion.get("clause"),
            suggestion.get("evidence_sentences"),
            suggestion.get("confidence"),
            json.dumps(suggestion.get("action_steps", [])),
            suggestion.get("explanation_evidence"),
            severity,
            "open",
            assigned_to,
            suggestion.get("page_reference"),
            suggestion.get("doc_reference"),
            flag_hash
        ))
        return cur.fetchone()[0]


# -------------------------
# Audit Actions
# -------------------------
def add_audit_action(flag_id: int, action_type: str, actor: str, comment: str = None):
    """
    Adds an audit trail entry for a compliance flag, linking to previous state.
    """
    with conn.cursor() as cur:
        # Get previous hash
        cur.execute("SELECT current_hash FROM audit_actions WHERE compliance_flag_id=%s ORDER BY id DESC LIMIT 1;", (flag_id,))
        row = cur.fetchone()
        if row and row[0]:
            previous_hash = row[0]
        else:
            cur.execute("SELECT hash FROM compliance_flags WHERE id=%s;", (flag_id,))
            prev = cur.fetchone()
            previous_hash = prev[0] if prev and prev[0] else "0"

        # Compute current hash
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{previous_hash}|{flag_id}|{action_type}|{actor}|{timestamp}|{comment or ''}"
        current_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        # Insert audit action
        cur.execute("""
            INSERT INTO audit_actions (compliance_flag_id, action_type, actor, timestamp, comment, previous_hash, current_hash)
            VALUES (%s,%s,%s,%s,%s,%s,%s);
        """, (flag_id, action_type, actor, timestamp, comment, previous_hash, current_hash))

        return current_hash

def insert_contract_deadline(contract_id, chunk_index, date, description, consequence):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO contract_deadlines (contract_id, chunk_index, date, deadline_description, consequence)
            VALUES (%s, %s, %s, %s, %s)
        """, (contract_id, chunk_index, date, description, consequence))
    conn.commit()

from app.db.connection import conn

def get_contract_id_from_db(file_name: str) -> int:
    """
    Fetch the contract ID (integer) from the database using the file name.
    Raises an error if the contract does not exist.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id FROM contracts WHERE file_name = %s
        """, (file_name,))
        result = cur.fetchone()
        if result:
            return result[0]  # the ID
        else:
            raise ValueError(f"Contract with file_name '{file_name}' not found.")

def insert_contract_chunk(text, embedding, metadata):
    query = """
        INSERT INTO contracts (file_name, page, chunk_index, text_chunk, embedding, 
                               jurisdiction, doc_type, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        metadata['file_name'],
        metadata['page'],
        metadata['chunk_index'],
        text,
        embedding,
        metadata.get('jurisdiction'),
        metadata.get('doc_type', 'contract'),
        json.dumps(metadata)
    )
    with conn.cursor() as cur:
     cur.execute(query, params)
     conn.commit()
from app.db.connection import conn



def store_clause_regulation_mapping(clause_id: int, mappings: list):
    """
    mappings: List of dicts with reg_name, article, status, explanation
    """
    query = """
        INSERT INTO clause_regulation_mapping
        (clause_id, reg_name, article, status, explanation)
        VALUES (%s, %s, %s, %s, %s)
    """
    cursor = conn.cursor()
    for m in mappings:
        cursor.execute(query, (clause_id, m.get("reg_name"), m.get("article"),
                               m.get("status"), m.get("explanation")))
    conn.commit()

from app.db.connection import conn
import numpy as np
from app.db.connection import conn
import ast  # safely parse string representations of lists

def get_contract_chunks(contract_id: int):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT id, text_chunk, embedding
            FROM contracts
            WHERE id = %s
        """, (contract_id,))
        rows = cursor.fetchall()

    chunks = []
    for r in rows:
        # Convert pgvector string to numpy array
        emb = np.array(ast.literal_eval(r[2]), dtype=np.float32)
        chunks.append({
            "id": r[0],
            "text_chunk": r[1],
            "embedding": emb
        })
    return chunks

def get_all_regulation_chunks():
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT id, text_chunk, embedding, metadata
            FROM regulations
        """)
        rows = cursor.fetchall()

    regs = []
    for r in rows:
        emb = np.array(ast.literal_eval(r[2]), dtype=np.float32)
        regs.append({
            "id": r[0],
            "text_chunk": r[1],
            "embedding": emb,
            "metadata": r[3]
        })
    return regs
