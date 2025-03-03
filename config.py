import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

class Config:
    """Centralized configuration settings for the RAG chatbot."""

    # Secret API Keys
    LANCHCHAIN_API_KEY = os.getenv("LANCHCHAIN_API_KEY")
    JINA_API_KEY = os.getenv("JINA_API_KEY")

    # Model Configurations
    OLLAMA_MODEL = "deepseek-r1:1.5b"
    EMBEDDING_MODEL = "jina-embeddings-v2-base-en"

    # Paths
    VECTOR_DB_PATH = "faiss_index"
    UPLOAD_FOLDER = "data"

    @staticmethod
    def display():
        """Display current configuration (excluding secrets)."""
        print(f"🔹 Ollama Model: {Config.OLLAMA_MODEL}")
        print(f"🔹 Embedding Model: {Config.EMBEDDING_MODEL}")
        print(f"🔹 Vector Store Path: {Config.VECTOR_DB_PATH}")
        print(f"🔹 Upload Folder: {Config.UPLOAD_FOLDER}")