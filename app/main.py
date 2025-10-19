from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  

from app.endpoints import ingest, search, rag, check_regulation, audit, compliance, tune
from app.endpoints import evaluate_contract 

app = FastAPI(title="Legal RAG API", version="6.0")
# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later to ["http://localhost:3000"]
    allow_credentials=True,#replace this * with actural frontend url when deploy/connect to real frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers from endpoints
app.include_router(ingest.router,  tags=["Ingest"])
app.include_router(search.router, tags=["Search"])
app.include_router(rag.router,  tags=["RAG"])
app.include_router(check_regulation.router, tags=["Check Regulation"])
app.include_router(audit.router,  tags=["Audit"])
app.include_router(compliance.router, tags=["Compliance"])
app.include_router(tune.router,  tags=["Tune"])
app.include_router(evaluate_contract.router, prefix="/rag", tags=["Contract Evaluation"])

@app.get("/")
def root():
    return {
        "message": "🚀 Legal RAG API with NER + Compliance Classification + Audit Trail",
        "endpoints": {
            "Upload PDF": "/ingest",
            "Semantic Search": "/search",
            "RAG Answer": "/rag",
            "Tune Threshold": "/tune",
            "Check Regulation": "/check-regulation",
            "Compliance Flags": "/compliance-flags",
            "Audit Action": "/audit-action"
        }
    }
