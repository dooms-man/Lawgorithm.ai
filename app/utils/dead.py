import re
import spacy
from nltk.tokenize import sent_tokenize
import nltk

nlp = spacy.load("en_core_web_sm")

# Ensure nltk punkt is available
def ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

ensure_nltk_data()

# --- Regex patterns ---
consequence_patterns = [
    r"(?:if|should|in case|failure to|breach).*?(?:penalt(?:y|ies)|interest|termination|damages?|forfeit|liable|compensation|legal action).*?\.?",
    r"(?:late|delayed|non[-\s]?performance).*?(?:will|shall|may).*?(?:result|lead).*?(?:penalt(?:y|ies)|interest|termination|damages?|forfeit|liable).*?\.?"
]

deadline_keywords = [
    r"(?:no later than|on or before|within|by)",
    r"(?:completion date|delivery date|due date)"
]

def extract_entities_and_deadlines(text: str):
    """
    Extract full sentences for deadlines, consequences, entities, and references.
    - Ensures sentences are merged properly (no mid-line splits)
    - Consequences captured with regex + NLP fallback
    - Deadlines captured as full sentences containing keywords
    """
    # --- 1. Merge lines first to avoid broken sentences ---
    merged_text = " ".join(line.strip() for line in text.splitlines() if line.strip())

    # --- 2. Sentence tokenization ---
    sentences = sent_tokenize(merged_text)

    # --- 3. Extract deadlines ---
    deadlines = []
    for sent in sentences:
        if any(re.search(kw, sent, re.IGNORECASE) for kw in deadline_keywords):
            deadlines.append({"sentence": sent.strip()})

    # --- 4. Extract consequences ---
    consequences = []
    for sent in sentences:
        for pattern in consequence_patterns:
            if re.search(pattern, sent, re.IGNORECASE):
                consequences.append({"sentence": sent.strip(), "method": "regex"})
                break

    # NLP fallback: check for penalty-related words
    doc = nlp(merged_text)
    for sent in doc.sents:
        if any(word in sent.text.lower() for word in [
            "penalty", "interest", "termination", "breach", "damages", "liable", "compensation"
        ]):
            if sent.text.strip() not in [c["sentence"] for c in consequences]:
                consequences.append({"sentence": sent.text.strip(), "method": "nlp"})

    # --- 5. Extract entities ---
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["DATE", "MONEY", "ORG", "PERCENT"]:
            entities.append({"text": ent.text, "label": ent.label_})
    if re.search(r"\bpenalt(y|ies)\b|\bfine\b", merged_text, re.IGNORECASE):
        entities.append({"text": merged_text, "label": "PENALTY"})
    if re.search(r"\bshall\b|\bmust\b", merged_text, re.IGNORECASE):
        entities.append({"text": merged_text, "label": "OBLIGATION"})

    # --- 6. Extract references ---
    references = []
    clause_pattern = re.findall(r"(Clause\s+\d+(\.\d+)*)", merged_text, re.IGNORECASE)
    section_pattern = re.findall(r"(Section\s+\d+(\.\d+)*)", merged_text, re.IGNORECASE)
    for c in clause_pattern:
        references.append(c[0])
    for s in section_pattern:
        references.append(s[0])

    return {
        "entities": entities,
        "references": references,
        "deadlines": deadlines,
        "consequences": consequences
    }
