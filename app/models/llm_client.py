# app/models/llm_client.py
from huggingface_hub import InferenceClient
from app.config import HF_TOKEN, LLM_MODEL
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


hf_client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

def query_llm(messages, max_tokens=300, temperature=0.7):
    response = hf_client.chat_completion(model=LLM_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature)
    return response.choices[0].message["content"]


# Load HuggingFace model
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, use_auth_token=HF_TOKEN)

# LLM pipeline for text generation
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  # GPU if available, else remove
    max_length=1024
)

def evaluate_clause_with_llm(clause_text: str, jurisdiction: str = None):
    """
    Sends a clause to LLM and returns suggested regulations, compliance status, and explanation.
    """
    prompt = f"""
    Analyze the following contract clause and suggest relevant regulations.
    Jurisdiction: {jurisdiction or 'General'}
    Clause: "{clause_text}"
    Return JSON in this format:
    [
        {{
            "reg_name": "string",
            "article": "string",
            "status": "compliant/violation",
            "explanation": "string"
        }}
    ]
    """
    output = llm_pipeline(prompt, max_length=1024, do_sample=False)
    text_output = output[0]['generated_text']

    # Parse JSON from LLM output
    import json
    try:
        return json.loads(text_output)
    except json.JSONDecodeError:
        return []
