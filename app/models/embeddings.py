# app/models/embeddings.py
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, models  # <-- add models here
model = SentenceTransformer("intfloat/multilingual-e5-large")

def encode_text(text: str):
    return model.encode(text).tolist()
# app/models/embeddings.py


# --- Load tokenizer + model from HuggingFace ---
word_embedding_model = models.Transformer("law-ai/InLegalBERT", max_seq_length=512)

# --- Use Mean Pooling for sentence embeddings ---
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

# --- Create SentenceTransformer model ---
model2 = SentenceTransformer(modules=[word_embedding_model, pooling_model])

print("✅ Law-specific SentenceTransformer loaded with MEAN pooling")