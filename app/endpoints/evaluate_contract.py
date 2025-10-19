from fastapi import APIRouter
import time, numpy as np, json, requests, os
from app.db.connection import conn
from app.config import HF_TOKEN, LLM_MODEL
from app.db.queries import get_contract_chunks, get_all_regulation_chunks, store_clause_regulation_mapping

# Config
from app.config import TOP_K_REGULATIONS, DISTANCE_THRESHOLD

# LLM config

from fastapi import APIRouter
import time, numpy as np, json, os
from app.db.connection import conn
from app.db.queries import get_contract_chunks, get_all_regulation_chunks, store_clause_regulation_mapping
from app.config import TOP_K_REGULATIONS, DISTANCE_THRESHOLD
from huggingface_hub import InferenceClient
from dotenv import load_dotenv


# --- Load regulations from DB ---
print("🔹 Loading regulation chunks from DB...")
regulations = get_all_regulation_chunks()
if regulations:
    reg_embeddings = np.array([r["embedding"] for r in regulations], dtype=np.float32)
    reg_texts = [r["text_chunk"] for r in regulations]
    print(f"✅ Loaded {len(regulations)} regulations with embeddings.\n")
else:
    reg_embeddings = np.zeros((0, 768), dtype=np.float32)
    reg_texts = []
    print("⚠️ No regulations found in DB. All clauses will be sent to LLM.\n")

# -------------------------------------------------------------------------
# LLM call via InferenceClient
# -------------------------------------------------------------------------
from app.models.llm_client import query_llm
router = APIRouter()

# -------------------------------------------------------------------------
# LLM call via hf_client
# -------------------------------------------------------------------------
def call_llm_for_explanation(clause_text: str, top_regs: list):
    try:
        prompt = (
            f"Clause: {clause_text}\n\n"
            f"Top Regulations: {json.dumps(top_regs, indent=2)}\n\n"
            "You are a legal compliance AI. Explain which regulations this clause likely maps to "
            "and whether there could be a compliance gap. Keep it short and clear."
        )
        messages = [
            {"role": "system", "content": "You are a concise, helpful legal compliance assistant."},
            {"role": "user", "content": prompt}
        ]
        # Use the query_llm function from llm_client.py
        return query_llm(messages, max_tokens=300, temperature=0.0)
    except Exception as e:
        print(f"❌ Zephyr API call failed: {e}")
        return "LLM unavailable. No explanation generated."

# -------------------------------------------------------------------------
# Main endpoint
# -------------------------------------------------------------------------
@router.post("/evaluate_contract/{contract_id}")
def evaluate_contract(contract_id: int):
    print(f"🚀 Evaluating Contract ID: {contract_id}")
    start_total = time.time()

    chunks = get_contract_chunks(contract_id)
    print(f"📄 Retrieved {len(chunks)} contract chunks.")

    total_mappings = 0

    for idx, chunk in enumerate(chunks):
        clause_text = chunk["text_chunk"].strip()
        if not clause_text:
            print(f"⚠️ Skipping empty clause ID {chunk['id']}")
            continue

        print(f"\n🔹 Processing Clause {idx+1}/{len(chunks)} (ID: {chunk['id']})")
        start_clause = time.time()

        # --- Step 1: Generate clause embedding (fixed) ---
        clause_embedding = np.array(
            chunk["embedding"] if chunk.get("embedding") is not None else np.zeros((768,), dtype=np.float32),
            dtype=np.float32
        )

        # --- Step 2: Semantic search if regulations exist ---
        top_regs = []
        if reg_embeddings.shape[0] > 0:
            sims = np.dot(reg_embeddings, clause_embedding) / (
                np.linalg.norm(reg_embeddings, axis=1) * (np.linalg.norm(clause_embedding) + 1e-10)
            )
            top_indices = np.argsort(sims)[::-1][:TOP_K_REGULATIONS]

            for i in top_indices:
                if sims[i] >= DISTANCE_THRESHOLD:
                    top_regs.append({
                        "reg_name": regulations[i]["metadata"].get("file_name"),
                        "article": regulations[i]["metadata"].get("chunk_index"),
                        "similarity": float(sims[i])
                    })

        # --- Step 3: Call Zephyr LLM if no strong matches or DB empty ---
        llm_explanation = ""
        if len(top_regs) == 0:
            print("   🤖 Sending clause to Zephyr LLM for regulation suggestion...")
            llm_explanation = call_llm_for_explanation(clause_text, top_regs)

        # --- Step 4: Store mapping ---
        mappings_to_store = top_regs or [{
            "reg_name": "LLM_Suggested",
            "article": "-",
            "status": "suggested",
            "explanation": llm_explanation
        }]
        store_clause_regulation_mapping(chunk["id"], mappings_to_store)
        total_mappings += len(mappings_to_store)
        print(f"   ✅ Clause stored in {time.time() - start_clause:.2f}s")

    print(f"\n🏁 Finished {len(chunks)} clauses in {time.time() - start_total:.2f}s")
    print(f"📊 Total mappings stored: {total_mappings}")

    return {
        "contract_id": contract_id,
        "clauses_evaluated": len(chunks),
        "mappings_stored": total_mappings,
        "total_time_seconds": time.time() - start_total,
        "response": llm_explanation
    }
