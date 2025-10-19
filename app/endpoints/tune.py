from fastapi import APIRouter
from app.db.connection import conn
from app.models.embeddings import model
from app.config import CONFIG_FILE
import numpy as np
import json

router = APIRouter()

@router.post("/tune-threshold")
def tune_threshold():
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_text FROM document_chunks LIMIT 20;")
        data = cur.fetchall()
    if not data:
        return {"error": "No data found in DB."}

    texts = [d[0] for d in data]
    embeddings = model.encode(texts, normalize_embeddings=True)
    sim_matrix = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(sim_matrix, 1.0)

    thresholds = np.arange(0.6, 0.9, 0.01)
    best_t, best_acc = 0, 0
    for t in thresholds:
        preds = (sim_matrix > t).astype(int)
        labels = (sim_matrix > 0.85).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t

    with open(CONFIG_FILE, "w") as f:
        json.dump({"DISTANCE_THRESHOLD": float(best_t)}, f)

    return {"message": "Auto-tuning completed.", "best_threshold": best_t, "accuracy_estimate": best_acc}
