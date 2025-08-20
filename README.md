# EduSmart AI Tutor

An AI-powered educational tutor system built with LangChain and Streamlit, featuring conversational memory, RAG (Retrieval-Augmented Generation) pipeline, and personalized learning experiences.

## 🎯 Project Overview

EduSmart AI Tutor is designed to provide personalized, interactive learning experiences for students. The system uses advanced AI technologies to:

- Retrieve relevant curriculum content using vector search
- Generate human-like, adaptive responses
- Remember past interactions for personalized learning
- Support multiple subjects with scalable architecture
- Ensure privacy and safety for educational use

## 🏗️ Architecture

### Core Components

1. **Document Processing** (`utils/document_processor.py`)
   - Loads and processes curriculum content (PDF, TXT files)
   - Splits documents into manageable chunks
   - Extracts metadata for better organization

2. **Vector Store Management** (`utils/vector_store.py`)
   - FAISS-based vector storage for efficient similarity search
   - HuggingFace embeddings for semantic understanding
   - Persistent storage and retrieval capabilities

3. **Conversation Memory** (`utils/conversation_memory.py`)
   - Tracks student learning patterns and preferences
   - Maintains conversation context
   - Builds personalized student profiles

4. **LLM Management** (`ai_tutor/llm_manager.py`)
   - HuggingFace model integration
   - Local model deployment for privacy
   - Fallback mechanisms for reliability

5. **RAG Pipeline** (`ai_tutor/rag_pipeline.py`)
   - ConversationalRetrievalChain implementation
   - Context-aware response generation
   - Source attribution and confidence scoring

6. **Main Tutor System** (`ai_tutor/tutor_system.py`)
   - Orchestrates all components
   - Session management
   - System monitoring and statistics

## 🚀 Features

### Educational Features
- **Personalized Learning**: Adapts to individual learning styles and preferences
- **Multi-Subject Support**: Mathematics, Science, English, History, Computer Science
- **Interactive Conversations**: Natural, engaging dialogue with memory retention
- **Source Attribution**: Shows curriculum sources for transparency
- **Confidence Scoring**: Indicates response reliability

### Technical Features
- **RAG Pipeline**: Retrieval-Augmented Generation for accurate responses
- **Vector Search**: FAISS-based semantic search through curriculum
- **Conversation Memory**: LangChain memory management
- **Local Deployment**: Privacy-focused local model execution
- **Scalable Architecture**: Easy to add new subjects and content

### UI/UX Features
- **Modern Interface**: Clean, educational-focused Streamlit UI
- **Real-time Chat**: Instant responses with typing indicators
- **Learning Suggestions**: Contextual learning recommendations
- **Session Management**: Save and restore learning sessions
- **Progress Tracking**: Monitor learning journey and statistics

## 📋 Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM (for local LLM)
- 5GB+ storage space

### Dependencies
```
streamlit==1.29.0
langchain==0.1.0
langchain-community==0.0.10
langchain-huggingface==0.0.1
sentence-transformers==2.2.2
faiss-cpu==1.7.4
huggingface-hub==0.19.4
transformers==4.36.2
torch==2.1.2
pypdf==3.17.4
python-dotenv==1.0.0
streamlit-chat==0.1.1
chromadb==0.4.18
tiktoken==0.5.2
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd edumart_ai_tutor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your preferences
   ```

5. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📚 Usage

### Starting the Application

1. Run `streamlit run streamlit_app.py`
2. Open your browser to `http://localhost:8501`
3. Wait for system initialization (first run may take a few minutes)

### Using the AI Tutor

1. **Select Subject**: Choose a subject focus from the sidebar (optional)
2. **Ask Questions**: Type your learning questions in the chat input
3. **Review Responses**: Get personalized explanations with sources
4. **Track Progress**: Monitor your learning journey in the sidebar

### Sample Questions

- "Can you explain algebra basics?"
- "How do Newton's laws work in real life?"
- "Help me understand photosynthesis"
- "What's the best way to solve quadratic equations?"
- "Explain the water cycle step by step"

## 🎓 Educational Philosophy

### Riya Malhotra - AI Character
- **Personality**: Warm, empathetic, student-centered
- **Background**: Former teacher with 10+ years in education
- **Approach**: Breaks down complex topics, provides examples, encourages learning
- **Goal**: Make AI tutoring feel as natural and helpful as human teaching

### Learning Approach
- **Adaptive**: Adjusts to student's learning style and pace
- **Interactive**: Encourages questions and exploration
- **Contextual**: Remembers previous conversations
- **Supportive**: Provides encouragement and positive reinforcement

## 🔧 Configuration

### Environment Variables (.env)
```
# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=microsoft/DialoGPT-medium

# Vector Store Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# App Configuration
MAX_CONVERSATION_HISTORY=10
```

### Adding Curriculum Content

1. Create a `curriculum_data` directory
2. Add PDF or TXT files with educational content
3. Restart the application to process new content
4. Content will be automatically indexed and searchable

## 🧪 Testing

### Manual Testing
1. Start the application
2. Try various question types across different subjects
3. Test conversation memory by referencing previous questions
4. Verify source attribution and confidence scores

### Automated Testing (Future Enhancement)
- Unit tests for individual components
- Integration tests for the full pipeline
- Performance benchmarks for response times

## 🔒 Privacy & Safety

### Data Privacy
- **Local Processing**: All AI processing happens locally
- **No External APIs**: No data sent to external services (by default)
- **Session Storage**: Conversations stored locally only
- **User Control**: Users can clear sessions and data

### Educational Safety
- **Content Filtering**: Curriculum-based responses
- **Source Attribution**: Transparent information sources
- **Confidence Scoring**: Indicates response reliability
- **Fallback Mechanisms**: Graceful handling of edge cases

## 🚀 Scalability & Future Enhancements

### Subject Expansion
- Easy addition of new subjects
- Modular curriculum organization
- Subject-specific prompting strategies

### Technical Improvements
- GPU acceleration for faster responses
- Advanced embedding models
- Multi-modal support (images, videos)
- Real-time collaboration features

### Educational Features
- Progress tracking and analytics
- Adaptive difficulty adjustment
- Gamification elements
- Parent/teacher dashboards

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For the powerful RAG framework
- **Streamlit**: For the intuitive web interface
- **HuggingFace**: For the transformer models and embeddings
- **FAISS**: For efficient vector search capabilities

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the code comments for implementation details

---

**EduSmart AI Tutor** - Making personalized learning accessible through AI technology! 🎓✨
