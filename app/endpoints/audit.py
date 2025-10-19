from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.db.queries import add_audit_action
from app.db.connection import conn

router = APIRouter()

class AuditActionRequest(BaseModel):
    compliance_flag_id: int
    action_type: str  # approve / reject / comment
    actor: str
    comment: Optional[str] = None

@router.post("/audit-action")
def audit_action(req: AuditActionRequest):
    if req.action_type.lower() not in {"approve", "reject", "comment"}:
        raise HTTPException(status_code=400, detail="Invalid action_type")
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM compliance_flags WHERE id=%s;", (req.compliance_flag_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="compliance_flag not found")
    current_hash = add_audit_action(req.compliance_flag_id, req.action_type.lower(), req.actor, req.comment)
    return {"message": "Audit action recorded", "current_hash": current_hash}
