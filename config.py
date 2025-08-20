"""
Configuration settings for EduSmart AI Tutor
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
    GEMINI_API_KEY = os.getenv("GEMINI", "")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN", "")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJ", "GenAIwithGemini")
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Use Gemini as primary model for this run (can override via .env)
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # App Configuration
    APP_TITLE = os.getenv("APP_TITLE", "EduSmart AI Tutor")
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    
    # File Paths
    VECTOR_STORE_PATH = "vector_store"
    CURRICULUM_PATH = "curriculum_data"
    
    # Streamlit Configuration
    PAGE_TITLE = "EduSmart AI - Your Personal Learning Assistant"
    PAGE_ICON = "🎓"
    LAYOUT = "wide"