# app/utils/text_cleaning.py
import re
import fitz  # PyMuPDF

def clean_redundant_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"( \b\w+\b)( \1)+", r"\1", text)
    return text.strip()

def extract_text_chunks(pdf_path: str, chunk_size=500, overlap=50):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text().replace("\n", " ").strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = clean_redundant_text(text[start:end])
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "page": page_num + 1})
            start += chunk_size - overlap
    return chunks
