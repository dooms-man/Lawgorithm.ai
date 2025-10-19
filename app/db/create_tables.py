# app/db/create_tables.py
from app.db.connection import conn

def create_audit_tables():
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS compliance_flags (
            id SERIAL PRIMARY KEY,
            regulation_id TEXT,
            clause TEXT,
            evidence_sentences TEXT,
            confidence FLOAT,
            action_steps JSONB,
            explanation_evidence TEXT,
            severity TEXT,
            status TEXT DEFAULT 'open',
            assigned_to TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            page_reference TEXT,
            doc_reference TEXT,
            hash TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_actions (
            id SERIAL PRIMARY KEY,
            compliance_flag_id INT REFERENCES compliance_flags(id),
            action_type TEXT,
            actor TEXT,
            timestamp TIMESTAMP DEFAULT NOW(),
            comment TEXT,
            previous_hash TEXT,
            current_hash TEXT
        );
        """)
