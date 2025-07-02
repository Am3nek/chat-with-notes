"""
Vector store utilities for chunking text and managing embeddings.
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Tuple, Optional
import pickle
import os
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, TOP_K_RETRIEVAL


class VectorStore:
    """Manages text chunking, embeddings, and similarity search using FAISS."""
    
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str, source: str = "document") -> List[Dict]:
        """
        Split text into chunks with metadata using LangChain Document structure.
        
        Args:
            text: Input text to chunk
            source: Source identifier for the text
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Create a LangChain Document
        document = Document(page_content=text, metadata={"source": source})
        
        # Split the document
        chunks = self.text_splitter.split_documents([document])
        
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                'text': chunk.page_content,
                'source': source,
                'chunk_id': i,
                'char_count': len(chunk.page_content),
                'word_count': len(chunk.page_content.split()),
                'metadata': chunk.metadata
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return np.array(embeddings).astype('float32')
    
    def create_index(self, chunk_dicts: List[Dict]) -> None:
        """
        Create FAISS index from chunk dictionaries.
        
        Args:
            chunk_dicts: List of chunk dictionaries
        """
        if not chunk_dicts:
            raise ValueError("No chunks provided for indexing")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunk_dicts]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks = texts
        self.chunk_metadata = chunk_dicts
        
        print(f"Created FAISS index with {len(texts)} chunks")
    
    def search_similar(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """
        Search for similar chunks given a query.
        
        Args:
            query: Search query
            k: Number of similar chunks to retrieve
            
        Returns:
            List of similar chunks with metadata and scores
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, len(self.chunks))  # Ensure k doesn't exceed available chunks
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                result = {
                    'text': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'similarity_score': float(score)
                }
                results.append(result)
        
        return results
    
    def save_index(self, filepath: str) -> None:
        """Save the FAISS index and associated data."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save chunks and metadata
        data = {
            'chunks': self.chunks,
            'chunk_metadata': self.chunk_metadata,
            'dimension': self.dimension
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str) -> bool:
        """
        Load a previously saved FAISS index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load chunks and metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = data['chunks']
            self.chunk_metadata = data['chunk_metadata']
            self.dimension = data['dimension']
            
            print(f"Index loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics about the current vector store."""
        if self.index is None:
            return {"status": "No index created"}
        
        total_chars = sum(chunk['char_count'] for chunk in self.chunk_metadata)
        total_words = sum(chunk['word_count'] for chunk in self.chunk_metadata)
        
        return {
            "total_chunks": len(self.chunks),
            "total_characters": total_chars,
            "total_words": total_words,
            "average_chunk_size": total_chars // len(self.chunks) if self.chunks else 0,
            "index_dimension": self.dimension
        }