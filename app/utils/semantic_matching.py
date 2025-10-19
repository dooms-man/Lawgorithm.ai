from app.db.connection import conn
from app.config import DISTANCE_THRESHOLD

def find_top_regulations_by_embedding(clause_embedding, top_k=3, jurisdiction=None):
    """
    Finds top matching regulations using pgvector similarity.
    """
    sql = """
        SELECT id, text_chunk, embedding, jurisdiction
        FROM regulations
        WHERE (%s IS NULL OR jurisdiction = %s)
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    cursor = conn.cursor()
    cursor.execute(sql, (jurisdiction, jurisdiction, clause_embedding, top_k))
    results = cursor.fetchall()

    # Filter by distance threshold if needed
    filtered_results = []
    for row in results:
        similarity = row[2]  # <=> returns distance
        if similarity <= DISTANCE_THRESHOLD:
            filtered_results.append(row)
    return filtered_results
