"""
RAG (Retrieval-Augmented Generation) pipeline for the AI tutor
"""

import logging
from typing import Dict, Any, List, Optional

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from utils.vector_store import VectorStoreManager
from utils.conversation_memory import ConversationMemoryManager
from ai_tutor.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for educational content retrieval and generation"""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_manager: LLMManager,
        memory_manager: ConversationMemoryManager,
    ):
        self.vector_store_manager = vector_store_manager
        self.llm_manager = llm_manager
        self.memory_manager = memory_manager
        self.chain = None
        self._initialize_chain()

    def _initialize_chain(self):
        """Initialize the conversational retrieval chain"""
        try:
            # Create custom prompt template
            prompt_template = self._create_prompt_template()

            # Get retriever
            retriever = self.vector_store_manager.get_retriever(
                search_kwargs={"k": 4}
            )

            # Create conversational retrieval chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm_manager.llm,
                retriever=retriever,
                memory=self.memory_manager.memory,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                return_source_documents=True,
                verbose=True,
            )

            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {str(e)}")
            self._initialize_fallback_chain()

    def _initialize_fallback_chain(self):
        """Initialize a simple fallback chain"""
        try:
            logger.info("Initializing fallback RAG pipeline")

            prompt_template = PromptTemplate(
                input_variables=["question", "chat_history"],
                template=(
                    "You are Riya Malhotra, an AI tutor at EduSmart AI. "
                    "You're warm, empathetic, and focused on helping students learn.\n\n"
                    "Previous conversation:\n\n{chat_history}\n\n"
                    "Student question: {question}\n\n"
                    "Provide a helpful, educational response that encourages learning:"
                ),
            )

            self.chain = LLMChain(
                llm=self.llm_manager.llm,
                prompt=prompt_template,
                verbose=True,
            )

            logger.info("Fallback RAG pipeline initialized")
        except Exception as e:
            logger.error(f"Error initializing fallback pipeline: {str(e)}")
            raise

    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for the RAG chain"""
        template = """You are Riya Malhotra, a warm and empathetic AI tutor at EduSmart AI. You help students learn course content in an interactive, memory-aware, and human-like way.

Your personality:
- Warm, empathetic, and student-centered
- Highly vision-driven and focused on long-term impact
- Former teacher with 10+ years in education
- Believes AI tutors should feel as helpful and natural as real teachers
- Sharp in asking the right educational questions

RELEVANT CURRICULUM CONTENT:
{context}

CURRENT QUESTION: {question}

Instructions:
1. Use the curriculum content to provide accurate, educational responses
2. Adapt your teaching style to the student's learning preferences
3. Break down complex concepts into manageable parts
4. Provide examples and real-world applications
5. Ask follow-up questions to check understanding
6. Be encouraging and supportive
7. If the curriculum content doesn't contain relevant information, use your general knowledge but acknowledge the limitation

Respond as Riya would - warm, educational, and focused on the student's learning journey:"""

        return PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

    def query(self, question: str, subject_filter: Optional[str] = None) -> Dict[str, Any]:
        """Process a student query through the RAG pipeline"""
        try:
            # Get personalized context
            student_context = self.memory_manager.get_personalized_context()

            # Prepare input
            if hasattr(self.chain, "invoke"):
                # For ConversationalRetrievalChain
                # ConversationalRetrievalChain supports chat_history; pass it along and include student context in the question
                chat_history = self.memory_manager.get_memory_variables().get("chat_history", [])
                composed_question = f"STUDENT CONTEXT: {student_context}\n\nQUESTION: {question}"
                inputs = {
                    "question": composed_question,
                    "chat_history": chat_history,
                }
                result = self.chain.invoke(inputs)
            else:
                # For simple LLMChain fallback
                chat_history = self.memory_manager.get_conversation_context()
                result_text = self.chain.run(question=question, chat_history=chat_history)
                result = {"answer": result_text, "source_documents": []}

            # Extract response and sources
            answer = result.get("answer", "I'm sorry, I couldn't generate a response.")
            source_docs = result.get("source_documents", [])

            # Add interaction to memory
            metadata = {
                "subject": subject_filter,
                "num_sources": len(source_docs),
                "sources": [doc.metadata.get("source_file", "unknown") for doc in source_docs],
            }
            self.memory_manager.add_interaction(question, answer, metadata)

            response = {
                "answer": answer,
                "sources": self._format_sources(source_docs),
                "student_profile": self.memory_manager.get_student_profile_summary(),
                "confidence": self._calculate_confidence(source_docs),
            }

            logger.info(f"Processed query successfully with {len(source_docs)} sources")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Fallback: try generating a response without retrieval using the LLM manager
            try:
                fallback_answer = self.llm_manager.generate_response(question)
                return {
                    "answer": fallback_answer,
                    "sources": [],
                    "student_profile": self.memory_manager.get_student_profile_summary(),
                    "confidence": 0.3,
                }
            except Exception as inner_e:
                logger.error(f"Fallback generation failed: {str(inner_e)}")
                return {
                    "answer": "I apologize, but I'm having trouble processing your question right now. Could you please try rephrasing it?",
                    "sources": [],
                    "student_profile": self.memory_manager.get_student_profile_summary(),
                    "confidence": 0.0,
                }

    def _format_sources(self, source_docs: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for display"""
        formatted_sources = []
        for i, doc in enumerate(source_docs):
            content = doc.page_content
            preview = (content[:200] + "...") if len(content) > 200 else content
            formatted_sources.append(
                {
                    "id": i + 1,
                    "content": preview,
                    "metadata": {
                        "source_file": doc.metadata.get("source_file", "Unknown"),
                        "subject": doc.metadata.get("subject", "General"),
                        "topic": doc.metadata.get("topic", "N/A"),
                    },
                }
            )
        return formatted_sources

    def _calculate_confidence(self, source_docs: List[Document]) -> float:
        """Calculate confidence score based on retrieved documents"""
        if not source_docs:
            return 0.3  # Low confidence without sources

        # Simple confidence calculation based on number and quality of sources
        base_confidence = min(0.9, 0.5 + (len(source_docs) * 0.1))

        quality_bonus = 0.0
        for doc in source_docs:
            if doc.metadata.get("file_type") == "pdf":
                quality_bonus += 0.05
            if doc.metadata.get("subject") != "general":
                quality_bonus += 0.05

        return min(1.0, base_confidence + quality_bonus)

    def get_subject_suggestions(self, query: str) -> List[str]:
        """Get subject suggestions based on query"""
        try:
            query_lower = query.lower()
            subjects = []
            subject_keywords = {
                "mathematics": ["math", "algebra", "geometry", "calculus", "equation", "solve", "calculate"],
                "science": ["science", "physics", "chemistry", "biology", "experiment", "theory"],
                "english": ["english", "literature", "writing", "grammar", "essay", "reading"],
                "history": ["history", "historical", "past", "ancient", "war", "civilization"],
                "computer": ["computer", "programming", "code", "algorithm", "software"],
            }
            for subject, keywords in subject_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    subjects.append(subject)
            return subjects if subjects else ["general"]
        except Exception as e:
            logger.error(f"Error getting subject suggestions: {str(e)}")
            return ["general"]

    def update_retriever_settings(self, k: int = 4, score_threshold: float = 0.5):
        """Update retriever settings"""
        try:
            if hasattr(self.chain, "retriever"):
                self.chain.retriever.search_kwargs = {
                    "k": k,
                    "score_threshold": score_threshold,
                }
            logger.info(f"Updated retriever settings: k={k}, threshold={score_threshold}")
        except Exception as e:
            logger.error(f"Error updating retriever settings: {str(e)}")
