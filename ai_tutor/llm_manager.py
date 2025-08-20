"""
LLM management for the AI tutor system
"""
import logging
import os
from typing import Dict, Any
from langchain_community.llms import HuggingFacePipeline, HuggingFaceHub
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages Large Language Model operations"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", gemini_api_key: str = "", huggingface_api_token: str = "", use_local: bool = False):
        self.model_name = model_name
        self.gemini_api_key = gemini_api_key
        self.huggingface_api_token = huggingface_api_token
        self.use_local = use_local
        self.llm = None
        self.tokenizer = None
        self.provider = "unknown"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the language model"""
        try:
            # Prefer Hugging Face Hub when token is available or model looks like a HF repo id
            if self.huggingface_api_token and ("/" in self.model_name or "mistral" in self.model_name.lower()):
                self._initialize_huggingface_hub_model()
            elif self.use_local:
                self._initialize_local_model()
            elif self.gemini_api_key and "gemini" in self.model_name.lower():
                self._initialize_gemini_model()
            else:
                self._initialize_fallback_model()
                
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            # Fallback to a simpler model
            self._initialize_fallback_model()
    
    def _initialize_huggingface_hub_model(self):
        """Initialize model via Hugging Face Hub Inference API"""
        try:
            logger.info(f"Loading Hugging Face Hub model: {self.model_name}")
            self.llm = HuggingFaceHub(
                repo_id=self.model_name,
                huggingfacehub_api_token=self.huggingface_api_token,
                model_kwargs={
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                    "top_p": 0.9,
                    "repetition_penalty": 1.05
                }
            )
            self.provider = "huggingface_hub"
            logger.info("Hugging Face Hub model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Hugging Face Hub model: {str(e)}")
            raise
    
    def _initialize_local_model(self):
        """Initialize local Hugging Face model (CPU/GPU)"""
        try:
            logger.info(f"Loading local model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Create LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.provider = "local_hf_pipeline"
            logger.info("Local model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing local model: {str(e)}")
            raise
    
    def _initialize_gemini_model(self):
        """Initialize Google Gemini model"""
        try:
            logger.info(f"Loading Gemini model: {self.model_name}")
            
            # Set the API key as environment variable
            os.environ["GOOGLE_API_KEY"] = self.gemini_api_key
            
            # Create Gemini LLM (remove deprecated convert_system_message_to_human)
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.gemini_api_key,
                temperature=0.7
            )
            self.provider = "gemini"
            logger.info("Gemini model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise
    
    def _initialize_fallback_model(self):
        """Initialize a simple fallback model"""
        try:
            logger.info("Initializing fallback model")
            # Use a very simple model as fallback
            from langchain_community.llms import FakeListLLM
            
            responses = [
                "I understand you're asking about that topic. Let me help you learn step by step.",
                "That's a great question! Let me break it down for you.",
                "I can see you're working on this concept. Here's how I'd explain it:",
                "Let's explore this together. What specific part would you like to focus on?",
                "That's an interesting point. Let me provide some guidance on this."
            ]
            
            self.llm = FakeListLLM(responses=responses)
            self.provider = "fallback_fake_llm"
            logger.info("Fallback model initialized")
            
        except Exception as e:
            logger.error(f"Error initializing fallback model: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate a response using the LLM"""
        try:
            if not self.llm:
                raise ValueError("LLM not initialized")
            
            # Format prompt for educational context
            formatted_prompt = self._format_educational_prompt(prompt)
            
            # Generate response
            response = self.llm(formatted_prompt)
            
            # Clean and format response
            cleaned_response = self._clean_response(response, prompt)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Could you please rephrase your question?"
    
    def _format_educational_prompt(self, user_input: str) -> str:
        """Format prompt for educational context"""
        system_prompt = """You are Riya Malhotra, a warm and empathetic AI tutor at EduSmart AI. You help students learn in an interactive, personalized way.

Your personality:
- Warm, empathetic, and student-centered
- Focused on long-term learning impact
- Encouraging and supportive
- Breaks down complex topics into manageable parts
- Adapts explanations to student's learning style

Guidelines:
- Always be encouraging and positive
- Use simple, clear language
- Provide examples when explaining concepts
- Ask follow-up questions to check understanding
- Relate learning to real-world applications
- Remember that you're talking to a student who wants to learn

Student question: {user_input}

Respond as Riya would, being helpful, encouraging, and educational:"""
        
        return system_prompt.format(user_input=user_input)
    
    def _clean_response(self, response: str, original_prompt: str) -> str:
        """Clean and format the model response"""
        try:
            # Remove the original prompt if it appears in the response
            if original_prompt in response:
                response = response.replace(original_prompt, "").strip()
            
            # Remove common artifacts
            response = response.replace("<|endoftext|>", "")
            response = response.replace("</s>", "")
            response = response.replace("<s>", "")
            
            # Ensure response starts appropriately
            if not response:
                response = "I'd be happy to help you with that! Could you provide a bit more detail about what you'd like to learn?"
            
            # Limit response length
            sentences = response.split('. ')
            if len(sentences) > 4:
                response = '. '.join(sentences[:4]) + '.'
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}")
            return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'provider': self.provider,
            'use_local': self.use_local,
            'is_initialized': self.llm is not None,
            'tokenizer_available': self.tokenizer is not None
        }