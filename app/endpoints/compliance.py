from fastapi import APIRouter, HTTPException
from typing import Optional
from app.db.connection import conn

router = APIRouter()

@router.get("/compliance-flags")
def get_flags(status: Optional[str] = None, limit: int = 50):
    with conn.cursor() as cur:
        if status:
            cur.execute("SELECT id, regulation_id, clause, severity, status, created_at, updated_at FROM compliance_flags WHERE status=%s ORDER BY created_at DESC LIMIT %s;", (status, limit))
        else:
            cur.execute("SELECT id, regulation_id, clause, severity, status, created_at, updated_at FROM compliance_flags ORDER BY created_at DESC LIMIT %s;", (limit,))
        rows = cur.fetchall()
    return {"flags": [{"id": r[0], "regulation_id": r[1], "clause": r[2], "severity": r[3], "status": r[4], "created_at": r[5], "updated_at": r[6]} for r in rows]}

@router.get("/compliance-flag/{flag_id}")
def get_flag(flag_id: int):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM compliance_flags WHERE id=%s;", (flag_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Flag not found")
        cols = [desc[0] for desc in cur.description]
        flag = dict(zip(cols, row))
    return {"flag": flag}
