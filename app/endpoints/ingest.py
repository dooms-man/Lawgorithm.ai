from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
import logging
import os

# Import OCR + summarizer from repo root ocr package
# (ensure ocr/ is on PYTHONPATH or project root is package root)
from ocr.process_doc import process_pdf
from ocr.summarizer import run as summarizer_run

router = APIRouter()
logger = logging.getLogger("ingest")
logger.setLevel(logging.INFO)

@router.post("/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    max_pages: int = Form(3),
    dpi: int = Form(300),
    outdir: str = Form("outputs")
):
    """
    Upload a PDF, run OCR (process_pdf) then summarizer.run on the produced OCR JSON.
    Returns combined OCR JSON and summarization result.
    """
    uploads_dir = Path("uploads")
    outputs_dir = Path(outdir)

    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    unique_prefix = uuid.uuid4().hex
    saved_name = f"{unique_prefix}_{file.filename}"
    saved_path = uploads_dir / saved_name

    try:
        with saved_path.open("wb") as dst:
            shutil.copyfileobj(file.file, dst)

        logger.info(f"Saved upload to {saved_path}")

        # Run OCR -> returns JSON-like dict and writes outputs/<stem>.json
        ocr_result = process_pdf(str(saved_path), outdir=str(outputs_dir), max_pages=max_pages, dpi=dpi)

        # The process_pdf writes outputs/<stem>.json. Build path and call summarizer.
        ocr_json_path = outputs_dir / f"{Path(saved_path).stem}.json"
        if not ocr_json_path.exists():
            # If process_pdf returned dict but did not write file, try to use the dict to save then summarize
            try:
                import json
                with open(ocr_json_path, "w", encoding="utf-8") as f:
                    json.dump(ocr_result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Could not write ocr json to {ocr_json_path}: {e}")

        # Summarize
        summary_result = summarizer_run(str(ocr_json_path), outdir=str(outputs_dir))

        # Build a compact response (the full OCR JSON can be large)
        response = {
            "status": "success",
            "uploaded_filename": file.filename,
            "saved_path": str(saved_path),
            "ocr_json_path": str(ocr_json_path),
            "ocr": ocr_result,
            "summary": summary_result
        }
        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        logger.exception("Ingest processing failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            file.file.close()
        except Exception:
            pass