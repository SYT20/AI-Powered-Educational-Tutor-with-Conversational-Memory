"""
Streamlit web application for EduSmart AI Tutor
"""
import streamlit as st
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from ai_tutor.tutor_system import EduSmartAITutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    
    .ai-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    
    .student-profile {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .system-status {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'tutor_system' not in st.session_state:
        st.session_state.tutor_system = None
        st.session_state.system_initialized = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True
    
    if 'selected_subject' not in st.session_state:
        st.session_state.selected_subject = "All Subjects"

def initialize_tutor_system():
    """Initialize the AI tutor system"""
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing EduSmart AI Tutor... This may take a few moments."):
            try:
                st.session_state.tutor_system = EduSmartAITutor()
                success = st.session_state.tutor_system.initialize()
                
                if success:
                    st.session_state.system_initialized = True
                    st.session_state.current_session_id = st.session_state.tutor_system.start_new_session()
                    st.success("✅ AI Tutor initialized successfully!")
                    return True
                else:
                    st.error("❌ Failed to initialize AI Tutor system")
                    return False
                    
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
                logger.error(f"Initialization error: {str(e)}")
                return False
    
    return st.session_state.system_initialized

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 EduSmart AI Tutor</h1>
        <p>Your Personal Learning Assistant powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with controls and information"""
    with st.sidebar:
        st.header("🎯 Learning Controls")
        
        # Subject filter
        subjects = ["All Subjects", "Mathematics", "Science", "English", "History", "Computer Science"]
        st.session_state.selected_subject = st.selectbox(
            "Select Subject Focus:",
            subjects,
            index=subjects.index(st.session_state.selected_subject)
        )
        
        # Display options
        st.header("📊 Display Options")
        st.session_state.show_sources = st.checkbox("Show Sources", value=st.session_state.show_sources)
        
        # Session controls
        st.header("📝 Session Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Session"):
                if st.session_state.tutor_system:
                    st.session_state.current_session_id = st.session_state.tutor_system.start_new_session()
                    st.session_state.chat_history = []
                    st.success("New session started!")
                    st.rerun()
        
        with col2:
            if st.button("💾 Save Session"):
                if st.session_state.tutor_system:
                    success = st.session_state.tutor_system.end_session(save_session=True)
                    if success:
                        st.success("Session saved!")
                    else:
                        st.error("Failed to save session")
        
        # System status
        if st.session_state.system_initialized and st.session_state.tutor_system:
            st.header("🔧 System Status")
            status = st.session_state.tutor_system.get_system_status()
            success_rate = (status['stats']['successful_responses'] / max(1, status['stats']['total_queries']) * 100)
            
            st.markdown(f"""
            <div class="system-status">
                <strong>Session:</strong> {st.session_state.current_session_id or 'None'}<br>
                <strong>Queries:</strong> {status['stats']['total_queries']}<br>
                <strong>Success Rate:</strong> {success_rate:.1f}%<br>
                <strong>Documents:</strong> {status['stats']['documents_loaded']}
            </div>
            """, unsafe_allow_html=True)

def display_chat_interface():
    """Display the main chat interface"""
    st.header("💬 Chat with Riya Malhotra")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑‍🎓 You:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            
            elif message['type'] == 'ai':
                confidence = message.get('confidence', 0.5)
                confidence_class = 'confidence-high' if confidence > 0.7 else 'confidence-medium' if confidence > 0.4 else 'confidence-low'
                
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>🤖 Riya:</strong> <span class="{confidence_class}">({confidence:.1%} confidence)</span><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources if enabled
                if st.session_state.show_sources and message.get('sources'):
                    with st.expander(f"📚 Sources ({len(message['sources'])} found)"):
                        for source in message['sources']:
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {source['id']}:</strong> {source['metadata']['source_file']}<br>
                                <strong>Subject:</strong> {source['metadata']['subject']}<br>
                                <strong>Content:</strong> {source['content']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Display student profile if available
                if message.get('student_profile'):
                    with st.expander("👤 Your Learning Profile"):
                        st.markdown(f"""
                        <div class="student-profile">
                            {message['student_profile']}
                        </div>
                        """, unsafe_allow_html=True)

def handle_user_input():
    """Handle user input and generate AI response"""
    # Chat input
    user_input = st.chat_input("Ask me anything about your studies! 📚")
    
    if user_input and st.session_state.system_initialized:
        # Add user message to history
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Get subject filter
        subject_filter = None if st.session_state.selected_subject == "All Subjects" else st.session_state.selected_subject.lower()
        
        # Generate AI response
        with st.spinner("🤔 Riya is thinking..."):
            try:
                response = st.session_state.tutor_system.chat(user_input, subject_filter)
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    'type': 'ai',
                    'content': response['answer'],
                    'sources': response.get('sources', []),
                    'student_profile': response.get('student_profile', ''),
                    'confidence': response.get('confidence', 0.5),
                    'timestamp': datetime.now()
                })
                
                # Show learning suggestions
                if st.session_state.tutor_system:
                    suggestions = st.session_state.tutor_system.get_learning_suggestions(user_input)
                    if suggestions:
                        st.info("💡 **Learning Suggestions:** " + " | ".join(suggestions))
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                logger.error(f"Response generation error: {str(e)}")
        
        # Rerun to update the display
        st.rerun()

def display_welcome_message():
    """Display welcome message for new users"""
    if not st.session_state.chat_history:
        st.markdown("""
        ### 👋 Welcome to EduSmart AI!
        
        Hi! I'm **Riya Malhotra**, your AI tutor. I'm here to help you learn in a personalized, interactive way.
        
        **What I can help you with:**
        - 📚 Explaining complex concepts in simple terms
        - 🧮 Solving math problems step by step
        - 🔬 Understanding scientific principles
        - 📖 Improving reading comprehension
        - 💡 Providing real-world examples and applications
        
        **How to get started:**
        1. Choose a subject from the sidebar (optional)
        2. Ask me any question about your studies
        3. I'll provide personalized explanations and remember our conversation
        
        **Example questions you can ask:**
        - "Can you explain algebra basics?"
        - "How do Newton's laws work in real life?"
        - "Help me understand this reading passage"
        - "What's the best way to solve this math problem?"
        
        Go ahead and ask me anything! I'm excited to help you learn! 🚀
        """)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Initialize tutor system
    if not initialize_tutor_system():
        st.stop()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display welcome message or chat interface
        if not st.session_state.chat_history:
            display_welcome_message()
        else:
            display_chat_interface()
        
        # Handle user input
        handle_user_input()
    
    with col2:
        # Quick actions and tips
        st.header("🚀 Quick Actions")
        
        sample_questions = [
            "Explain algebra basics",
            "How do forces work?",
            "Help with reading comprehension",
            "What is photosynthesis?",
            "Solve: 2x + 5 = 15"
        ]
        
        st.write("**Try these sample questions:**")
        for question in sample_questions:
            if st.button(f"💬 {question}", key=f"sample_{question}"):
                # Simulate user input
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': question,
                    'timestamp': datetime.now()
                })
                
                # Generate response
                subject_filter = None if st.session_state.selected_subject == "All Subjects" else st.session_state.selected_subject.lower()
                
                with st.spinner("🤔 Generating response..."):
                    try:
                        response = st.session_state.tutor_system.chat(question, subject_filter)
                        
                        st.session_state.chat_history.append({
                            'type': 'ai',
                            'content': response['answer'],
                            'sources': response.get('sources', []),
                            'student_profile': response.get('student_profile', ''),
                            'confidence': response.get('confidence', 0.5),
                            'timestamp': datetime.now()
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                st.rerun()
        
        # Learning tips
        st.header("💡 Learning Tips")
        st.markdown("""
        **For better learning:**
        - Ask specific questions
        - Request examples
        - Ask for step-by-step explanations
        - Connect topics to real life
        - Practice with different problems
        """)

if __name__ == "__main__":
    main()