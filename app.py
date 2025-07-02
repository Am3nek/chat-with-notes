"""
Streamlit app for Chat with Your Notes - RAG system with Mistral API.
"""
import streamlit as st
import os
from io import BytesIO
import time
from datetime import datetime

# Import our modules
from pdf_utils import PDFExtractor
from vector_store import VectorStore
from qa_chain import MistralQAChain, RAGPipeline
from config import (
    APP_TITLE, MAX_FILE_SIZE_MB, MISTRAL_API_KEY, 
    MISTRAL_MODEL, TOP_K_RETRIEVAL
)


def initialize_session_state():
    """Initialize session state variables."""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'pdf_metadata' not in st.session_state:
        st.session_state.pdf_metadata = None


def setup_page():
    """Configure Streamlit page."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 " + APP_TITLE)
    st.markdown("*Upload a PDF document and ask questions about its content using Mistral AI*")


def check_api_key():
    """Check if Mistral API key is configured."""
    if not MISTRAL_API_KEY or MISTRAL_API_KEY == "your-mistral-api-key-here":
        st.error("🔑 Mistral API key not configured!")
        st.markdown("""
        Please set your Mistral API key in one of the following ways:
        1. Create a `.env` file with: `MISTRAL_API_KEY=your_actual_key_here`
        2. Set the environment variable: `export MISTRAL_API_KEY=your_actual_key_here`
        3. Edit `config.py` directly (not recommended for production)
        
        Get your API key from: https://console.mistral.ai/
        """)
        return False
    return True


def process_pdf(uploaded_file):
    """Process uploaded PDF file."""
    if uploaded_file is None:
        return False
    
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
        return False
    
    try:
        with st.spinner("📄 Extracting text from PDF..."):
            # Extract text
            pdf_extractor = PDFExtractor()
            pdf_bytes = BytesIO(uploaded_file.read())
            
            # Get metadata
            metadata = pdf_extractor.get_pdf_metadata(pdf_bytes)
            st.session_state.pdf_metadata = metadata
            
            # Extract and clean text
            text = pdf_extractor.extract_and_clean(pdf_bytes)
            
            if not text:
                st.error("Could not extract text from PDF. Please ensure it's a text-based PDF.")
                return False
            
            st.success(f"✅ Extracted text from {metadata['num_pages']} pages")
        
        with st.spinner("🔤 Creating text chunks and embeddings..."):
            # Initialize vector store
            vector_store = VectorStore()
            
            # Chunk the text
            chunks = vector_store.chunk_text(text, source=uploaded_file.name)
            
            # Create embeddings and index
            vector_store.create_index(chunks)
            
            # Store in session state
            st.session_state.vector_store = vector_store
            
            # Initialize QA chain
            qa_chain = MistralQAChain()
            
            # Test API connection
            if not qa_chain.test_connection():
                st.error("Failed to connect to Mistral API. Please check your API key.")
                return False
            
            # Create RAG pipeline
            st.session_state.rag_pipeline = RAGPipeline(vector_store, qa_chain)
            
            st.success(f"✅ Created {len(chunks)} text chunks and embeddings")
            st.session_state.document_processed = True
            
            return True
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False


def display_document_info():
    """Display information about the processed document."""
    if st.session_state.pdf_metadata and st.session_state.vector_store:
        with st.expander("📊 Document Information", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**PDF Metadata:**")
                st.write(f"• Pages: {st.session_state.pdf_metadata['num_pages']}")
                st.write(f"• Title: {st.session_state.pdf_metadata['title']}")
                st.write(f"• Author: {st.session_state.pdf_metadata['author']}")
            
            with col2:
                stats = st.session_state.vector_store.get_stats()
                st.write("**Processing Stats:**")
                st.write(f"• Total chunks: {stats['total_chunks']}")
                st.write(f"• Total words: {stats['total_words']:,}")
                st.write(f"• Avg chunk size: {stats['average_chunk_size']} chars")


def display_chat_interface():
    """Display the chat interface."""
    st.subheader("💬 Ask Questions About Your Document")
    
    # Display chat history
    for i, (question, answer, timestamp) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🙋 You ({timestamp}):**")
            st.markdown(question)
            st.markdown(f"**🤖 Assistant:**")
            st.markdown(answer)
            st.markdown("---")
    
    # Question input
    with st.form(key="question_form"):
        question = st.text_area(
            "Ask a question about your document:",
            placeholder="e.g., What are the main topics covered in this document?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("Ask Question")
        with col2:
            if st.form_submit_button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    if submit_button and question.strip():
        process_question(question)


def process_question(question):
    """Process user question and generate response."""
    with st.spinner("🤔 Thinking..."):
        try:
            # Get response from RAG pipeline
            result = st.session_state.rag_pipeline.query(question, k=TOP_K_RETRIEVAL)
            
            if result['success']:
                response = result['response']
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Add to chat history
                st.session_state.chat_history.append((question, response, timestamp))
                
                # Display response details in expander
                with st.expander("🔍 Response Details", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Response Metadata:**")
                        st.write(f"• Model: {result.get('model', 'Unknown')}")
                        st.write(f"• Chunks used: {result.get('num_chunks_used', 0)}")
                        if 'usage' in result:
                            st.write(f"• Tokens used: {result['usage']['total_tokens']}")
                    
                    with col2:
                        st.write("**Retrieval Scores:**")
                        for i, score in enumerate(result.get('retrieval_scores', []), 1):
                            st.write(f"• Chunk {i}: {score:.3f}")
                
                # Show retrieved chunks
                if result.get('retrieved_chunks'):
                    with st.expander("📄 Retrieved Text Chunks", expanded=False):
                        for i, chunk in enumerate(result['retrieved_chunks'], 1):
                            st.write(f"**Chunk {i} (Score: {chunk['similarity_score']:.3f}):**")
                            st.write(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                            st.write("---")
                
                st.rerun()
                
            else:
                st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


def main():
    """Main application function."""
    # Initialize
    initialize_session_state()
    setup_page()
    
    # Check API key
    if not check_api_key():
        return
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF", type="primary"):
                if process_pdf(uploaded_file):
                    st.success("Document processed successfully!")
                    st.rerun()
        
        # Settings
        st.header("⚙️ Settings")
        st.write(f"**Model:** {MISTRAL_MODEL}")
        st.write(f"**Retrieval chunks:** {TOP_K_RETRIEVAL}")
        
        # Clear session button
        if st.button("🗑️ Clear Session"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    # Main content area
    if not st.session_state.document_processed:
        st.info("👆 Please upload and process a PDF document to start asking questions.")
        
        # Show example questions
        st.subheader("💡 Example Questions You Can Ask:")
        st.markdown("""
        - What are the main topics covered in this document?
        - Can you summarize the key findings?
        - What methodology was used in this research?
        - What are the conclusions or recommendations?
        - Explain the concept of [specific term] mentioned in the document
        - What are the limitations discussed?
        """)
        
    else:
        # Show document info
        display_document_info()
        
        # Show chat interface
        display_chat_interface()


if __name__ == "__main__":
    main()