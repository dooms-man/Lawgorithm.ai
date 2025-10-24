# process_doc.py (patched for richer JSON, logic untouched)
import os
import argparse
import logging
import fitz  # PyMuPDF
from pathlib import Path
from paddleocr import PaddleOCR
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import camelot
import json
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------
# OCR Engines
# -------------------------------
paddle_en = None
try:
    paddle_en = PaddleOCR(lang="en", det=True, rec=True, cls=True)
except Exception as e:
    logging.warning(f"PaddleOCR not fully available: {e}")

tess_config = "--psm 6"

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# -------------------------------
# Language Heuristic
# -------------------------------
def detect_language_heuristic(text: str) -> str:
    if not text.strip():
        return "unknown"
    mal_count = sum([0x0D00 <= ord(c) <= 0x0D7F for c in text])
    ratio = mal_count / len(text)
    if ratio > 0.4:
        return "ml"
    elif ratio > 0.1:
        return "hybrid"
    else:
        return "en"

# -------------------------------
# Preprocessing for Malayalam
# -------------------------------
def preprocess_for_malayalam(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return str(image_path)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = thresh.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out_path = str(image_path).replace(".png", "_mlproc.png")
    cv2.imwrite(out_path, deskewed)
    return out_path

# -------------------------------
# OCR Runners
# -------------------------------
def run_paddle(image_path):
    if not paddle_en:
        return None
    try:
        result = paddle_en.ocr(image_path, cls=True)
        lines = []
        for block in result[0]:
            if len(block) >= 2:
                text, conf = block[1]
                box = block[0]
                lines.append({
                    "text": text,
                    "conf": float(conf),
                    "source": "paddle",
                    "box": box
                })
        return lines
    except Exception as e:
        logging.warning(f"Paddle failed: {e}")
        return None

def run_tesseract(image_path, lang="eng"):
    try:
        text = pytesseract.image_to_string(Image.open(image_path), config=tess_config, lang=lang)
        return [{
            "text": line,
            "conf": None,
            "source": f"tesseract-{lang}",
            "box": None
        } for line in text.splitlines() if line.strip()]
    except Exception as e:
        logging.warning(f"Tesseract failed: {e}")
        return None

def run_trocr(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return [{
            "text": text,
            "conf": None,
            "source": "trocr",
            "box": None
        }]
    except Exception as e:
        logging.warning(f"TrOCR failed: {e}")
        return None

def extract_tables(pdf_path, max_pages=3):
    try:
        tables = camelot.read_pdf(pdf_path, pages=f"1-{max_pages}", flavor="lattice")
        extracted = []
        for i, table in enumerate(tables):
            extracted.append({
                "page": table.page,
                "table_id": i,
                "data": table.df.to_dict()
            })
        return extracted
    except Exception as e:
        logging.warning(f"Camelot table extraction failed: {e}")
        return []

# -------------------------------
# Main Processor
# -------------------------------
def process_pdf(pdf_path, outdir="outputs", max_pages=2, dpi=300):
    pdf_path = Path(pdf_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Processing file: {pdf_path}")
    doc = fitz.open(pdf_path)

    result_json = {
        "file": str(pdf_path),
        "metadata": {"pages": len(doc)},
        "pages": [],
        "tables": []
    }

    tables = extract_tables(str(pdf_path), max_pages=max_pages)
    result_json["tables"] = tables

    for page_num in range(min(max_pages, len(doc))):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_path = outdir / f"{pdf_path.stem}_p{page_num+1}.png"
        pix.save(img_path)

        logging.info(f"Page {page_num+1}: OCR {img_path}")

        rough_text = pytesseract.image_to_string(Image.open(img_path), config="--psm 6", lang="eng+mal")
        lang = detect_language_heuristic(rough_text)
        logging.info(f"Heuristic language detection: {lang}")

        lines = None
        if lang == "ml":
            preproc_img = preprocess_for_malayalam(img_path)
            lines = run_tesseract(preproc_img, lang="mal")
        elif lang == "hybrid":
            lines = run_paddle(str(img_path)) or run_tesseract(str(img_path), lang="eng")
        else:
            lines = run_paddle(str(img_path)) or run_tesseract(str(img_path), lang="eng")

        if not lines:
            lines = run_trocr(str(img_path))

        chunked_text = " ".join([ln["text"] for ln in lines]) if lines else ""

        result_json["pages"].append({
            "page_number": page_num + 1,
            "language": lang,
            "chunked_text": chunked_text,
            "lines": lines if lines else []
        })

    out_path = outdir / f"{pdf_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved processed JSON → {out_path}")
    return result_json

# -------------------------------
# CLI
# -------------------------------
def main(args):
    process_pdf(args.file, outdir=args.outdir, max_pages=args.max_pages, dpi=args.dpi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="PDF file path")
    parser.add_argument("--outdir", default="outputs", help="output folder for processed JSON")
    parser.add_argument("--max-pages", type=int, default=3, help="max pages to process")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF->image conversion")
    args = parser.parse_args()
    main(args)