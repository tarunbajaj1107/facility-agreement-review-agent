import os

# Configuration & Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "precedents_db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_precedents")

# Ensure directories exist
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model Settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# Fast, free open-source LLM via Groq (Get a free key at https://console.groq.com)
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
