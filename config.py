"""
Configuration file for the Chat with Your Notes app.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Mistral API Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "your-mistral-api-key-here")
MISTRAL_MODEL = "mistral-large-latest"  # or "mistral-medium", "mistral-small"

# Vector Store Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 3

# Embedding Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformers model

# App Configuration
APP_TITLE = "Chat with Your Notes"
MAX_FILE_SIZE_MB = 10