"""
Vector store management for RAG pipeline
"""
import os
import pickle
import logging
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations for document retrieval"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create a new vector store from documents"""
        try:
            if not documents:
                raise ValueError("No documents provided for vector store creation")
            
            logger.info(f"Creating vector store with {len(documents)} documents")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info("Vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing vector store"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            if not documents:
                logger.warning("No documents to add")
                return
            
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def save_vector_store(self, path: str) -> None:
        """Save vector store to disk"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            os.makedirs(path, exist_ok=True)
            self.vector_store.save_local(path)
            
            # Save metadata
            metadata = {
                'embedding_model': self.embedding_model,
                'num_documents': self.vector_store.index.ntotal
            }
            
            with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Vector store saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(self, path: str) -> Optional[FAISS]:
        """Load vector store from disk"""
        try:
            if not os.path.exists(path):
                logger.warning(f"Vector store path does not exist: {path}")
                return None
            
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata if available
            metadata_path = os.path.join(path, 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info(f"Loaded vector store with {metadata.get('num_documents', 'unknown')} documents")
            
            logger.info(f"Vector store loaded from {path}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def similarity_search(self, query: str, k: int = 4, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            if filter_dict:
                # FAISS doesn't support metadata filtering directly
                # We'll implement a simple post-filtering approach
                results = self.vector_store.similarity_search(query, k=k*2)
                filtered_results = []
                
                for doc in results:
                    if all(doc.metadata.get(key) == value for key, value in filter_dict.items()):
                        filtered_results.append(doc)
                        if len(filtered_results) >= k:
                            break
                
                return filtered_results[:k]
            else:
                return self.vector_store.similarity_search(query, k=k)
                
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            return self.vector_store.similarity_search_with_score(query, k=k)
            
        except Exception as e:
            logger.error(f"Error performing similarity search with score: {str(e)}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """Get a retriever object for use in chains"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)