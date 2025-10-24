# summarizer.py
import os
import argparse
import json
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline

# -------------------------------
# Init summarizer
# -------------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# -------------------------------
# Functions
# -------------------------------
def load_text_with_meta(json_path):
    """Extract all lines of text with page/line references."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for page in data.get("pages", []):
        for idx, line in enumerate(page.get("lines", [])):
            if "text" in line and line["text"].strip():
                lines.append({
                    "text": line["text"].strip(),
                    "page_number": page["page_number"],
                    "line_number": idx + 1
                })
    return lines

def make_summary(full_text):
    if not full_text.strip():
        return "No text available for summarization."
    # Summarize in chunks (BART has max length ~1024 tokens)
    inputs = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
    outputs = summarizer(inputs, max_length=150, min_length=40, do_sample=False)
    return " ".join([o['summary_text'] for o in outputs])

def make_word_stats(lines, top_k=10):
    # Flatten text
    words = " ".join([l["text"] for l in lines]).lower().split()
    counts = Counter(words)
    # Top K words
    top_words = counts.most_common(top_k)

    # Map back to pages/lines
    word_refs = {}
    for word, count in top_words:
        refs = []
        for l in lines:
            if word in l["text"].lower():
                refs.append({
                    "page": l["page_number"],
                    "line": l["line_number"],
                    "context": l["text"]
                })
        word_refs[word] = {
            "count": count,
            "occurrences": refs
        }
    return word_refs

def make_wordcloud(lines, outdir, fname="wordcloud.png"):
    text = " ".join([l["text"] for l in lines])
    wc = WordCloud(width=1200, height=800, background_color="white").generate(text)
    out_path = os.path.join(outdir, fname)
    wc.to_file(out_path)
    return out_path

def run(json_path, outdir="outputs", top_k=10):
    os.makedirs(outdir, exist_ok=True)
    lines = load_text_with_meta(json_path)
    full_text = " ".join([l["text"] for l in lines])

    summary = make_summary(full_text)
    word_stats = make_word_stats(lines, top_k=top_k)
    wc_path = make_wordcloud(lines, outdir)

    result = {
        "source_json": json_path,
        "summary": summary,
        "top_words": word_stats,
        "wordcloud_image": wc_path
    }

    out_path = os.path.join(outdir, os.path.basename(json_path).replace(".json", "_sum.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved summarization JSON → {out_path}")
    print(f"✅ Wordcloud saved → {wc_path}")
    return result

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to OCR JSON file")
    parser.add_argument("--outdir", default="outputs", help="Folder for results")
    parser.add_argument("--top-k", type=int, default=10, help="Top K frequent words")
    args = parser.parse_args()

    run(args.json, outdir=args.outdir, top_k=args.top_k)
