import os
import json

# ----------------------------------------------------------------------
# 🗄️ Database configuration
# ----------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5433))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASS = os.getenv("DB_PASS", "ragpass")

# ----------------------------------------------------------------------
# 🤖 LLM Configuration (Hugging Face)
# ----------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "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")
LLM_MODEL = os.getenv("LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta")

# ----------------------------------------------------------------------
# 📄 Chunking / Similarity Config
# ----------------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ----------------------------------------------------------------------
# ⚙️ Load optional values from config.json
# ----------------------------------------------------------------------
CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        try:
            config_data = json.load(f)
        except json.JSONDecodeError:
            config_data = {}
else:
    config_data = {}

# ----------------------------------------------------------------------
# 📊 Similarity search parameters
# ----------------------------------------------------------------------
# Default values if not found in config.json
DISTANCE_THRESHOLD = config_data.get("DISTANCE_THRESHOLD", 0.8)
TOP_K_REGULATIONS = config_data.get("TOP_K_REGULATIONS", 3)

# ----------------------------------------------------------------------
# ✅ Debug summary
# ----------------------------------------------------------------------
print(f"✅ Loaded config:")
print(f"   → DB: {DB_NAME}@{DB_HOST}:{DB_PORT}")
print(f"   → LLM_MODEL: {LLM_MODEL}")
print(f"   → DISTANCE_THRESHOLD: {DISTANCE_THRESHOLD}")
print(f"   → TOP_K_REGULATIONS: {TOP_K_REGULATIONS}")
