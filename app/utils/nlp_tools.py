# app/utils/nlp_tools.py
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def classify_compliance_area(text: str) -> str:
    text_lower = text.lower()
    if any(k in text_lower for k in ["data protection", "gdpr", "personal data", "privacy"]):
        return "Privacy"
    elif any(k in text_lower for k in ["cybersecurity", "access control", "encryption", "security"]):
        return "Security"
    elif any(k in text_lower for k in ["report", "audit", "disclosure", "filing", "notify"]):
        return "Reporting"
    elif any(k in text_lower for k in ["anti-money laundering", "bribery", "fraud"]):
        return "AML/Integrity"
    else:
        return "General Compliance"

def extract_entities(text: str):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "LAW", "GPE"]:
            entities.append({"text": ent.text, "label": "REGULATION_NAME"})
    if re.search(r"\bmust\b|\bshall\b|\bshould\b|\brequired to\b|\bprohibited\b", text, re.IGNORECASE):
        entities.append({"text": text, "label": "OBLIGATION"})
    compliance_area = classify_compliance_area(text)
    entities.append({"text": text, "label": f"COMPLIANCE_AREA_{compliance_area.upper()}"} )
    if re.search(r"\bpenalt(y|ies)\b|\bfine\b|\bimprisonment\b", text, re.IGNORECASE):
        entities.append({"text": text, "label": "PENALTY"})
    return entities

def detect_references(text: str):
    refs = []
    clause_pattern = re.findall(r"(Clause\s+\d+(\.\d+)*)", text, re.IGNORECASE)
    section_pattern = re.findall(r"(Section\s+\d+(\.\d+)*)", text, re.IGNORECASE)
    refs.extend([c[0] for c in clause_pattern])
    refs.extend([s[0] for s in section_pattern])
    return refs
