"""
Main AI Tutor System that orchestrates all components
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import Config
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.conversation_memory import ConversationMemoryManager
from ai_tutor.llm_manager import LLMManager
from ai_tutor.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EduSmartAITutor:
    """Main AI Tutor System"""

    def __init__(self):
        self.config = Config()
        self.is_initialized = False
        self.document_processor = None
        self.vector_store_manager = None
        self.memory_manager = None
        self.llm_manager = None
        self.rag_pipeline = None
        self.current_session_id = None
        self.system_stats = {
            "total_queries": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "documents_loaded": 0,
        }

    def initialize(self) -> bool:
        """Initialize the AI tutor system"""
        try:
            logger.info("Initializing EduSmart AI Tutor System...")

            # Document Processor
            self.document_processor = DocumentProcessor(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
            )
            logger.info("Document processor initialized")

            # Vector Store Manager
            self.vector_store_manager = VectorStoreManager(
                embedding_model=self.config.EMBEDDING_MODEL
            )
            logger.info("Vector store manager initialized")

            # Conversation Memory
            self.memory_manager = ConversationMemoryManager(
                max_history=self.config.MAX_CONVERSATION_HISTORY
            )
            logger.info("Conversation memory initialized")

            # LLM Manager
            self.llm_manager = LLMManager(
                model_name=self.config.LLM_MODEL,
                gemini_api_key=self.config.GEMINI_API_KEY,
                huggingface_api_token=self.config.HUGGINGFACE_API_TOKEN,
                use_local=False,
            )
            logger.info("LLM manager initialized")

            # Load or create vector store
            self._setup_vector_store()

            # RAG Pipeline
            self.rag_pipeline = RAGPipeline(
                vector_store_manager=self.vector_store_manager,
                llm_manager=self.llm_manager,
                memory_manager=self.memory_manager,
            )
            logger.info("RAG pipeline initialized")

            self.is_initialized = True
            logger.info("EduSmart AI Tutor System initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Error initializing AI tutor system: {str(e)}")
            self.is_initialized = False
            return False

    def _setup_vector_store(self):
        """Setup vector store with curriculum data"""
        try:
            vector_store_path = self.config.VECTOR_STORE_PATH
            if os.path.exists(vector_store_path):
                logger.info("Loading existing vector store...")
                vs = self.vector_store_manager.load_vector_store(vector_store_path)
                if vs:
                    logger.info("Vector store loaded successfully")
                    return

            logger.info("Creating new vector store with curriculum content...")
            documents = self._load_curriculum_documents()
            if not documents:
                logger.warning("No curriculum documents found, using sample content")
                documents = self.document_processor.create_sample_curriculum()

            chunks = self.document_processor.split_documents(documents)
            self.system_stats["documents_loaded"] = len(chunks)

            self.vector_store_manager.create_vector_store(chunks)
            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store_manager.save_vector_store(vector_store_path)
            logger.info(f"Vector store created with {len(chunks)} document chunks")
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise

    def _load_curriculum_documents(self) -> List:
        """Load curriculum documents from the curriculum directory"""
        try:
            curriculum_path = self.config.CURRICULUM_PATH
            if not os.path.exists(curriculum_path):
                logger.info(f"Curriculum directory not found: {curriculum_path}")
                return []

            file_paths = []
            for root, dirs, files in os.walk(curriculum_path):
                for file in files:
                    if file.lower().endswith((".pdf", ".txt")):
                        file_paths.append(os.path.join(root, file))

            if not file_paths:
                logger.info("No curriculum files found")
                return []

            documents = self.document_processor.load_documents(file_paths)
            logger.info(f"Loaded {len(documents)} curriculum documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading curriculum documents: {str(e)}")
            return []

    def start_new_session(self, session_id: Optional[str] = None) -> str:
        """Start a new learning session"""
        try:
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_session_id = session_id
            if self.memory_manager:
                self.memory_manager.clear_memory()
            logger.info(f"Started new session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error starting new session: {str(e)}")
            return "default_session"

    def end_session(self, save_session: bool = True) -> bool:
        """End the current learning session"""
        try:
            if save_session and self.memory_manager and self.current_session_id:
                session_file = f"sessions/{self.current_session_id}.json"
                os.makedirs("sessions", exist_ok=True)
                self.memory_manager.save_session(session_file)
                logger.info(f"Session saved: {session_file}")
                self.current_session_id = None
            return True
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")
            return False

    def chat(self, message: str, subject_filter: Optional[str] = None) -> Dict[str, Any]:
        """Process a chat message from the student"""
        try:
            if not self.is_initialized:
                return {
                    "answer": "I'm sorry, but the AI tutor system is not properly initialized. Please try again later.",
                    "sources": [],
                    "student_profile": "",
                    "confidence": 0.0,
                    "error": "System not initialized",
                }

            self.system_stats["total_queries"] += 1
            response = self.rag_pipeline.query(message, subject_filter)

            if response.get("answer"):
                self.system_stats["successful_responses"] += 1
            else:
                self.system_stats["failed_responses"] += 1

            response["session_id"] = self.current_session_id
            response["query_count"] = self.system_stats["total_queries"]
            return response
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            self.system_stats["failed_responses"] += 1
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try rephrasing it or ask something else.",
                "sources": [],
                "student_profile": self.memory_manager.get_student_profile_summary() if self.memory_manager else "",
                "confidence": 0.0,
                "error": str(e),
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        return {
            "is_initialized": self.is_initialized,
            "current_session": self.current_session_id,
            "stats": self.system_stats.copy(),
            "components": {
                "document_processor": self.document_processor is not None,
                "vector_store_manager": self.vector_store_manager is not None,
                "memory_manager": self.memory_manager is not None,
                "llm_manager": self.llm_manager is not None,
                "rag_pipeline": self.rag_pipeline is not None,
            },
            "model_info": self.llm_manager.get_model_info() if self.llm_manager else {},
        }

    def add_curriculum_content(self, file_paths: List[str]) -> bool:
        """Add new curriculum content to the system"""
        try:
            if not self.is_initialized:
                logger.error("System not initialized")
                return False

            documents = self.document_processor.load_documents(file_paths)
            if not documents:
                logger.warning("No documents loaded")
                return False

            chunks = self.document_processor.split_documents(documents)
            self.vector_store_manager.add_documents(chunks)
            self.vector_store_manager.save_vector_store(self.config.VECTOR_STORE_PATH)
            self.system_stats["documents_loaded"] += len(chunks)
            logger.info(f"Added {len(chunks)} new document chunks")
            return True
        except Exception as e:
            logger.error(f"Error adding curriculum content: {str(e)}")
            return False

    def get_learning_suggestions(self, student_query: str) -> List[str]:
        """Get learning suggestions based on student query"""
        try:
            if not self.rag_pipeline:
                return ["Let's start with the basics of your topic of interest!"]

            subjects = self.rag_pipeline.get_subject_suggestions(student_query)
            suggestions = []
            for subject in subjects:
                if subject == "mathematics":
                    suggestions.extend(
                        [
                            "Would you like to practice solving equations?",
                            "Let's explore some real-world math applications!",
                            "How about we work through some step-by-step examples?",
                        ]
                    )
                elif subject == "science":
                    suggestions.extend(
                        [
                            "Want to learn about scientific experiments?",
                            "Let's explore how science applies to everyday life!",
                            "Would you like to understand the theory behind this concept?",
                        ]
                    )
                elif subject == "english":
                    suggestions.extend(
                        [
                            "Let's practice reading comprehension together!",
                            "Would you like help with writing techniques?",
                            "How about we analyze some interesting texts?",
                        ]
                    )
                else:
                    suggestions.extend(
                        [
                            "Let's break this topic down into smaller parts!",
                            "Would you like to see some examples?",
                            "How about we explore this concept step by step?",
                        ]
                    )
            return suggestions[:3]
        except Exception as e:
            logger.error(f"Error getting learning suggestions: {str(e)}")
            return ["I'm here to help you learn! What would you like to explore?"]
