"""
Conversation memory management for personalized learning
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class ConversationMemoryManager:
    """Manages conversation memory and learning context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.memory = ConversationBufferWindowMemory(
            k=max_history,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.student_profile = {
            'learning_style': 'adaptive',
            'difficulty_preference': 'medium',
            'subjects_of_interest': [],
            'common_mistakes': [],
            'strengths': [],
            'session_count': 0,
            'last_session': None
        }
        self.conversation_history = []
    
    def add_interaction(self, human_input: str, ai_response: str, metadata: Optional[Dict] = None):
        """Add a new interaction to memory"""
        try:
            # Add to LangChain memory
            self.memory.chat_memory.add_user_message(human_input)
            self.memory.chat_memory.add_ai_message(ai_response)
            
            # Add to detailed history with metadata
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'human_input': human_input,
                'ai_response': ai_response,
                'metadata': metadata or {}
            }
            
            self.conversation_history.append(interaction)
            
            # Keep only recent interactions
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            # Update student profile
            self._update_student_profile(human_input, ai_response, metadata)
            
            logger.info("Added interaction to conversation memory")
            
        except Exception as e:
            logger.error(f"Error adding interaction to memory: {str(e)}")
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context"""
        try:
            if not self.conversation_history:
                return "This is the beginning of our conversation."
            
            recent_interactions = self.conversation_history[-3:]  # Last 3 interactions
            context_parts = []
            
            for interaction in recent_interactions:
                context_parts.append(f"Student: {interaction['human_input']}")
                context_parts.append(f"Tutor: {interaction['ai_response'][:200]}...")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return ""
    
    def get_student_profile_summary(self) -> str:
        """Get a summary of the student's learning profile"""
        try:
            profile = self.student_profile
            summary_parts = []

            # Safely coerce list items to strings and drop Nones
            def _clean(items):
                return [str(x) for x in items if x]
            
            subjects = _clean(profile.get('subjects_of_interest', []))
            if subjects:
                summary_parts.append(f"Interested in: {', '.join(subjects)}")
            
            strengths = _clean(profile.get('strengths', []))
            if strengths:
                summary_parts.append(f"Strengths: {', '.join(strengths)}")
            
            mistakes = _clean(profile.get('common_mistakes', []))
            if mistakes:
                summary_parts.append(f"Areas for improvement: {', '.join(mistakes)}")
            
            summary_parts.append(f"Learning style: {profile.get('learning_style', 'adaptive')}")
            summary_parts.append(f"Preferred difficulty: {profile.get('difficulty_preference', 'medium')}")
            summary_parts.append(f"Sessions completed: {profile.get('session_count', 0)}")
            
            return " | ".join(summary_parts) if summary_parts else "New student profile"
            
        except Exception as e:
            logger.error(f"Error getting student profile summary: {str(e)}")
            return "Profile unavailable"
    
    def _update_student_profile(self, human_input: str, ai_response: str, metadata: Optional[Dict]):
        """Update student profile based on interaction"""
        try:
            # Extract subject from metadata or input
            if metadata and 'subject' in metadata:
                subject = metadata['subject']
                if subject and isinstance(subject, str):
                    if subject not in self.student_profile['subjects_of_interest']:
                        self.student_profile['subjects_of_interest'].append(subject)
            
            # Analyze input for learning patterns
            input_lower = human_input.lower()
            
            # Detect difficulty preferences
            if any(word in input_lower for word in ['easy', 'simple', 'basic']):
                self.student_profile['difficulty_preference'] = 'easy'
            elif any(word in input_lower for word in ['hard', 'difficult', 'challenging', 'advanced']):
                self.student_profile['difficulty_preference'] = 'hard'
            
            # Detect learning style indicators
            if any(word in input_lower for word in ['example', 'show me', 'demonstrate']):
                self.student_profile['learning_style'] = 'visual'
            elif any(word in input_lower for word in ['explain', 'why', 'how']):
                self.student_profile['learning_style'] = 'analytical'
            elif any(word in input_lower for word in ['practice', 'try', 'do']):
                self.student_profile['learning_style'] = 'hands-on'
            
            # Update session info
            self.student_profile['session_count'] += 1
            self.student_profile['last_session'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error updating student profile: {str(e)}")
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for use in chains"""
        return self.memory.load_memory_variables({})
    
    def clear_memory(self):
        """Clear conversation memory"""
        try:
            self.memory.clear()
            self.conversation_history = []
            logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def save_session(self, filepath: str):
        """Save conversation session to file"""
        try:
            session_data = {
                'student_profile': self.student_profile,
                'conversation_history': self.conversation_history,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving session: {str(e)}")
    
    def load_session(self, filepath: str):
        """Load conversation session from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.student_profile = session_data.get('student_profile', self.student_profile)
            self.conversation_history = session_data.get('conversation_history', [])
            
            # Restore LangChain memory
            self.memory.clear()
            for interaction in self.conversation_history[-self.max_history:]:
                self.memory.chat_memory.add_user_message(interaction['human_input'])
                self.memory.chat_memory.add_ai_message(interaction['ai_response'])
            
            logger.info(f"Session loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading session: {str(e)}")
    
    def get_personalized_context(self) -> str:
        """Get personalized context for the AI tutor"""
        try:
            context_parts = [
                "STUDENT CONTEXT:",
                f"Learning Style: {self.student_profile['learning_style']}",
                f"Difficulty Preference: {self.student_profile['difficulty_preference']}",
                f"Session Count: {self.student_profile['session_count']}"
            ]
            
            if self.student_profile['subjects_of_interest']:
                context_parts.append(f"Interested Subjects: {', '.join(self.student_profile['subjects_of_interest'])}")
            
            if self.student_profile['strengths']:
                context_parts.append(f"Strengths: {', '.join(self.student_profile['strengths'])}")
            
            if self.student_profile['common_mistakes']:
                context_parts.append(f"Areas to Focus: {', '.join(self.student_profile['common_mistakes'])}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting personalized context: {str(e)}")
            return "STUDENT CONTEXT: New student"