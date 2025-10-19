# app/utils/prioritization.py

# Example: priority mapping for jurisdictions
priority_order = {
    "local": 1,
    "international": 2,
    "company_policy": 3
}

def classify_jurisdiction(chunk_text, jurisdiction=None):
    if jurisdiction and re.search(rf"\b{jurisdiction}\b", chunk_text, re.IGNORECASE):
        return "local", 3
    elif re.search(r"\b(EU|Europe|European Union|GDPR|ISO)\b", chunk_text, re.IGNORECASE):
        return "international", 2
    else:
        return "company_policy", 1
