from app.db.connection import conn
from app.db.queries import store_compliance_flag
from app.models.embeddings import model
from app.models.llm_client import hf_client, LLM_MODEL
from app.config import DISTANCE_THRESHOLD
from sentence_transformers import util
import numpy as np
import re

def detect_gaps_for_regulation(reg_chunks, top_k=5):
    """
    Compare each regulation chunk against all contracts/internal_compliances,
    detect gaps (geo, retention, encryption), store flags with LLM explanation.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id, text_chunk, metadata, embedding FROM contracts;")

        contracts = cur.fetchall()

    suggestions = []

    for reg_chunk in reg_chunks:
        reg_text = reg_chunk['text']
        reg_page = reg_chunk['page']
        reg_emb = model.encode(reg_text)

        for contract in contracts:
            contract_id, contract_text, contract_meta, contract_emb_db = contract
            contract_emb = np.array(contract_emb_db)
            sim = float(util.cos_sim(reg_emb, contract_emb).numpy()[0][0])

            if sim < DISTANCE_THRESHOLD:
                continue

            geo_flag = retention_flag = encryption_flag = False
            action_steps = []

            # Geo/location check
            if re.search(r"\b(EU|Europe|European Union)\b", reg_text, re.IGNORECASE):
                if re.search(r"\b(us|usa|us-east-1|us-west-2|india|ap-south-1)\b", contract_text, re.IGNORECASE):
                    geo_flag = True
                    action_steps.append(
                        "Move your database to an EU region or implement lawful transfer mechanisms (SCCs)."
                    )

            # Retention check
            retention_match = re.search(r"retain(ed|ion).*?(\d+)\s+years?", contract_text, re.IGNORECASE)
            if retention_match:
                years = int(retention_match.group(2))
                if re.search(r"\bmax(?:imum)?\s*1\s+year\b", reg_text, re.IGNORECASE):
                    if years > 1:
                        retention_flag = True
                        action_steps.append(
                            "Adjust retention to meet regulatory maximum (purge/archive older records)."
                        )

            # Encryption check
            if re.search(r"\bencrypted\b|\bTLS\b|\bencryption at rest\b", reg_text, re.IGNORECASE):
                if not re.search(r"\bencrypted\b|\bTLS\b|\bencryption at rest\b", contract_text, re.IGNORECASE):
                    encryption_flag = True
                    action_steps.append(
                        "Enable encryption at rest and TLS in transit; update config & document keys."
                    )

            # Store suggestion if gap exists
            if geo_flag or retention_flag or encryption_flag:
                # LLM explanation
                messages = [
                    {"role": "system", "content": "You are a compliance assistant explaining gaps clearly."},
                    {"role": "user", "content": f"Regulation: {reg_text}\nContract: {contract_text}\nSuggested Actions: {action_steps}"}
                ]
                try:
                    explanation_resp = hf_client.chat_completion(
                        model=LLM_MODEL, messages=messages, max_tokens=180, temperature=0.2
                    )
                    explanation_text = explanation_resp.choices[0].message["content"]
                except Exception:
                    explanation_text = "LLM explanation unavailable."

                suggestion = {
                    "regulation_id": reg_chunk.get("file_name", "UNKNOWN") + "_REG",
                    "clause": reg_text,
                    "evidence_sentences": contract_text,
                    "confidence": sim,
                    "action_steps": action_steps,
                    "explanation_evidence": explanation_text,
                    "page_reference": reg_page,
                    "doc_reference": contract_meta.get("file_name", "UNKNOWN")
                }

                flag_id = store_compliance_flag(suggestion, severity="high")
                suggestion["flag_id"] = flag_id
                suggestions.append(suggestion)

    return suggestions
