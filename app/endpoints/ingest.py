from fastapi import APIRouter, UploadFile, File, Form
from app.utils.text_cleaning import extract_text_chunks
from app.utils.dead import extract_entities_and_deadlines
from app.models.embeddings import model
from app.db.queries import (
    insert_chunk,
    insert_regulation_chunk,
    insert_contract_deadline,
    insert_contract_chunk,
    get_contract_id_from_db
)
from app.endpoints.gap_detection import detect_gaps_for_regulation
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

router = APIRouter()

@router.get("/")
def root():
    return {
        "message": "🚀 Legal RAG API - Regulation & Compliance Ingest Endpoint",
        "info": "Upload PDFs, extract chunks, store embeddings, extract deadlines/consequences, and find compliance gaps."
    }


@router.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    doc_type: str = Form(...),  # "regulation" | "internal_compliance" | "contract"
    jurisdiction: str = Form(None)
):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    if doc_type not in ["regulation", "internal_compliance", "contract"]:
        return {"error": "Invalid document type."}

    pdf_path = f"./{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    chunks = extract_text_chunks(pdf_path, CHUNK_SIZE, CHUNK_OVERLAP)
    processed_count = 0
    all_deadlines, all_consequences = [], []

    # Initialize contract_id (used in metadata)
    contract_id = None
    if doc_type == "contract":
        contract_id = get_contract_id_from_db(file.filename)

    for idx, chunk in enumerate(chunks):
        text = chunk["text"]
        embedding = model.encode(text).tolist()

        info = extract_entities_and_deadlines(text)
        deadlines = info.get("deadlines", [])
        consequences = info.get("consequences", [])
        entities = info.get("entities", [])
        references = info.get("references", [])

        metadata = {
            "file_name": file.filename,
            "page": chunk["page"],
            "chunk_index": idx,
            "doc_type": doc_type,
            "jurisdiction": jurisdiction,
            "deadlines": deadlines,
            "consequences": consequences,
            "entities": entities,
            "references": references,
            "contractid": contract_id
        }

        # Insert into main table
        insert_chunk(text, embedding, metadata)
        processed_count += 1
        all_deadlines.extend(deadlines)
        all_consequences.extend(consequences)

        # Handle contract-specific inserts
        if doc_type == "contract" and deadlines:
            for d in deadlines:
                insert_contract_chunk(text, embedding, metadata)
                insert_contract_deadline(
                    contract_id=contract_id,
                    chunk_index=idx,
                    date=d.get("date"),
                    description=d.get("sentence"),
                    consequence=None
                )

        # Handle regulation-specific inserts
        if doc_type == "regulation":
            if not jurisdiction:
                return {"error": "Jurisdiction is required for regulation documents."}
            insert_regulation_chunk(text, embedding, metadata)

    # Gap detection for regulation
    flags_generated = 0
    if doc_type == "regulation":
        suggestions = detect_gaps_for_regulation(chunks)
        flags_generated = len(suggestions)

    return {
        "status": "success",
        "file_name": file.filename,
        "doc_type": doc_type,
        "chunks_processed": processed_count,
        "deadlines_extracted": all_deadlines,
        "consequences_extracted": all_consequences,
        "compliance_flags_generated": flags_generated,
        "jurisdiction": jurisdiction,
        "contractid": contract_id
    }
