"""
Document processing utilities for curriculum content
"""
import os
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and processing for curriculum content"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from various file formats"""
        documents = []
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_extension == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                else:
                    logger.warning(f"Unsupported file format: {file_extension}")
                    continue
                
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        'source_file': os.path.basename(file_path),
                        'file_type': file_extension,
                        'subject': self._extract_subject_from_filename(file_path)
                    })
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return []
    
    def _extract_subject_from_filename(self, file_path: str) -> str:
        """Extract subject from filename"""
        filename = os.path.basename(file_path).lower()
        
        subjects = {
            'math': ['math', 'mathematics', 'algebra', 'geometry', 'calculus'],
            'science': ['science', 'physics', 'chemistry', 'biology'],
            'history': ['history', 'social', 'studies'],
            'english': ['english', 'literature', 'language', 'writing'],
            'computer': ['computer', 'programming', 'coding', 'cs']
        }
        
        for subject, keywords in subjects.items():
            if any(keyword in filename for keyword in keywords):
                return subject
        
        return 'general'
    
    def create_sample_curriculum(self) -> List[Document]:
        """Create sample curriculum content for demonstration"""
        sample_content = [
            {
                'content': """
                Mathematics - Algebra Basics
                
                Algebra is a branch of mathematics that uses symbols and letters to represent numbers and quantities in formulas and equations.
                
                Key Concepts:
                1. Variables: Letters like x, y, z that represent unknown values
                2. Constants: Fixed numbers like 5, -3, 0.5
                3. Expressions: Combinations of variables and constants like 2x + 3
                4. Equations: Mathematical statements that show equality like 2x + 3 = 7
                
                Solving Linear Equations:
                - Isolate the variable on one side
                - Use inverse operations
                - Check your solution by substituting back
                
                Example: Solve 2x + 3 = 7
                Step 1: Subtract 3 from both sides: 2x = 4
                Step 2: Divide both sides by 2: x = 2
                Step 3: Check: 2(2) + 3 = 4 + 3 = 7 ✓
                """,
                'subject': 'mathematics',
                'topic': 'algebra_basics'
            },
            {
                'content': """
                Science - Introduction to Physics
                
                Physics is the study of matter, energy, and their interactions in the universe.
                
                Fundamental Concepts:
                1. Motion: How objects move through space and time
                2. Force: A push or pull that can change an object's motion
                3. Energy: The ability to do work or cause change
                4. Matter: Anything that has mass and takes up space
                
                Newton's Laws of Motion:
                1. First Law (Inertia): An object at rest stays at rest, and an object in motion stays in motion, unless acted upon by an external force.
                2. Second Law: Force equals mass times acceleration (F = ma)
                3. Third Law: For every action, there is an equal and opposite reaction.
                
                Applications:
                - Understanding how cars brake and accelerate
                - Explaining why we wear seatbelts
                - Rocket propulsion and space travel
                """,
                'subject': 'science',
                'topic': 'physics_intro'
            },
            {
                'content': """
                English Literature - Reading Comprehension Strategies
                
                Reading comprehension is the ability to understand, analyze, and interpret written text.
                
                Key Strategies:
                1. Preview: Look at titles, headings, and images before reading
                2. Predict: Make educated guesses about what will happen
                3. Question: Ask yourself questions while reading
                4. Summarize: Identify main ideas and key details
                5. Connect: Relate the text to your own experiences
                
                Types of Questions:
                - Literal: Information directly stated in the text
                - Inferential: Reading between the lines
                - Critical: Evaluating and analyzing the author's purpose
                
                Active Reading Techniques:
                - Highlight or underline important information
                - Take notes in the margins
                - Create mental images of what you're reading
                - Pause periodically to reflect on what you've learned
                """,
                'subject': 'english',
                'topic': 'reading_comprehension'
            }
        ]
        
        documents = []
        for item in sample_content:
            doc = Document(
                page_content=item['content'],
                metadata={
                    'subject': item['subject'],
                    'topic': item['topic'],
                    'source_file': 'sample_curriculum',
                    'file_type': 'generated'
                }
            )
            documents.append(doc)
        
        return documents