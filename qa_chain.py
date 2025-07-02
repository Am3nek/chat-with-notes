"""
QA Chain for retrieval and response generation using Mistral API via LangChain.
"""
from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from typing import List, Dict, Optional
from config import MISTRAL_API_KEY, MISTRAL_MODEL, TOP_K_RETRIEVAL


class MistralQAChain:
    """Question-Answering chain using LangChain Mistral integration."""
    
    def __init__(self, api_key: str = MISTRAL_API_KEY, model: str = MISTRAL_MODEL):
        """
        Initialize the Mistral QA Chain.
        
        Args:
            api_key: Mistral API key
            model: Mistral model name
        """
        if not api_key or api_key == "your-mistral-api-key-here":
            raise ValueError("Please set your Mistral API key in config.py or environment variables")
        
        self.llm = ChatMistralAI(
            api_key=api_key,
            model=model,
            temperature=0.1,  # Low temperature for more focused responses
            max_tokens=1000,  # Adjust based on your needs
        )
        self.model = model
    
    def create_context_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        Create a context-aware prompt from retrieved chunks.
        
        Args:
            query: User's question
            retrieved_chunks: List of retrieved chunks with metadata
            
        Returns:
            Formatted prompt string
        """
        if not retrieved_chunks:
            return f"""Please answer the following question based on your general knowledge:

Question: {query}

Please provide a helpful and accurate response."""
        
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"Context {i} (similarity: {chunk['similarity_score']:.3f}):\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an intelligent assistant helping users understand their documents. Use the provided context to answer the user's question accurately and comprehensively.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Base your answer primarily on the provided context
2. If the context doesn't contain enough information to fully answer the question, clearly state what information is missing
3. Be specific and cite relevant parts of the context when possible
4. If the question cannot be answered from the context, say so clearly
5. Provide a clear, well-structured response that directly addresses the question

ANSWER:"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> Dict[str, any]:
        """
        Generate response using LangChain Mistral integration.
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Create system and human messages
            system_message = SystemMessage(content="You are an intelligent assistant helping users understand their documents. Provide accurate, helpful responses based on the given context.")
            human_message = HumanMessage(content=prompt)
            
            messages = [system_message, human_message]
            
            # Generate response with token tracking
            response = self.llm.invoke(messages)
            
            # Extract token usage if available
            usage_info = {}
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                token_usage = response.response_metadata['token_usage']
                usage_info = {
                    "prompt_tokens": token_usage.get('prompt_tokens', 0),
                    "completion_tokens": token_usage.get('completion_tokens', 0),
                    "total_tokens": token_usage.get('total_tokens', 0)
                }
            
            return {
                "success": True,
                "response": response.content,
                "model": self.model,
                "usage": usage_info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"Sorry, I encountered an error while processing your question: {str(e)}"
            }
    
    def answer_question(self, query: str, retrieved_chunks: List[Dict]) -> Dict[str, any]:
        """
        Complete QA pipeline: create prompt and generate response.
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved chunks from vector store
            
        Returns:
            Dictionary containing the complete response and metadata
        """
        # Create context-aware prompt
        prompt = self.create_context_prompt(query, retrieved_chunks)
        
        # Generate response
        result = self.generate_response(prompt)
        
        # Add additional metadata
        result.update({
            "query": query,
            "num_chunks_used": len(retrieved_chunks),
            "chunk_sources": [chunk.get('metadata', {}).get('source', 'unknown') 
                            for chunk in retrieved_chunks]
        })
        
        return result
    
    def test_connection(self) -> bool:
        """
        Test the connection to Mistral API via LangChain.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_message = HumanMessage(content="Hello, this is a test.")
            response = self.llm.invoke([test_message])
            return True
        except Exception as e:
            print(f"Mistral API connection test failed: {e}")
            return False


class RAGPipeline:
    """Complete RAG pipeline combining vector store and QA chain."""
    
    def __init__(self, vector_store, qa_chain: MistralQAChain):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: VectorStore instance
            qa_chain: MistralQAChain instance
        """
        self.vector_store = vector_store
        self.qa_chain = qa_chain
    
    def query(self, question: str, k: int = TOP_K_RETRIEVAL) -> Dict[str, any]:
        """
        Process a question through the complete RAG pipeline.
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            
        Returns:
            Complete response with metadata
        """
        try:
            # Retrieve relevant chunks
            retrieved_chunks = self.vector_store.search_similar(question, k=k)
            
            # Generate answer
            result = self.qa_chain.answer_question(question, retrieved_chunks)
            
            # Add retrieval metadata
            result.update({
                "retrieved_chunks": retrieved_chunks,
                "retrieval_scores": [chunk['similarity_score'] for chunk in retrieved_chunks]
            })
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "query": question
            }