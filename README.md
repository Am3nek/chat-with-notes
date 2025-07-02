# Chat with Your Notes 📚

A powerful Streamlit application that allows you to upload PDF documents and chat with them using Mistral AI's large language models. Built with LangChain for robust document processing and retrieval-augmented generation (RAG).

## Features ✨

* **PDF Upload & Processing**: Extract text from PDF documents with multiple extraction methods
* **Intelligent Chunking**: Split documents into semantic chunks for better retrieval
* **Vector Search**: Use FAISS for fast similarity search with sentence transformers
* **Mistral AI Integration**: Powered by Mistral's latest language models via LangChain
* **Interactive Chat**: Natural conversation interface with chat history
* **Document Overview**: Auto-generate comprehensive document summaries
* **Advanced Features**: Detailed response analysis and chunk inspection
* **Modular Architecture**: Clean, maintainable code structure

## Quick Start 🚀

### 1. Installation

```bash
# Clone or download the project files
git clone https://github.com/Am3nek/chat-with-notes.git
cd chat-with-notes

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Mistral API key:

```bash
MISTRAL_API_KEY=your_actual_mistral_api_key_here
```

Get your API key from: https://console.mistral.ai/

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide 📖

### Basic Workflow

1. **Upload PDF**: Click "Choose a PDF file" in the sidebar
2. **Process Document**: Click "Process PDF" to extract and index the content
3. **Ask Questions**: Use the chat interface to ask questions about your document

### Quick Actions

* **📋 Get Document Overview**: Generate a comprehensive summary
* **🔍 Key Topics**: Extract main themes and topics
* **📝 Summary**: Get a structured summary of the document

### Advanced Features

* **Advanced Mode**: Check the "Advanced" option for detailed response information
* **Chunk Inspection**: View the exact text chunks used to generate responses
* **Relevance Scores**: See how relevant each chunk is to your question

## Architecture 🏗️

The application follows a modular architecture:

```
├── app.py              # Main Streamlit application
├── config.py           # Configuration and settings
├── pdf_utils.py        # PDF text extraction utilities
├── vector_store.py     # Text chunking and vector operations
├── qa_chain.py         # LangChain QA pipeline with Mistral
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create from .env.example)
└── .env.example       # Template for environment variables
```

### Key Components

* **PDFExtractor**: Robust PDF text extraction with fallback methods
* **VectorStore**: Text chunking and FAISS-based similarity search
* **QAChain**: Retrieval-augmented generation using Mistral AI
* **StreamlitUI**: Interactive chat interface with document management

## Technical Details 🔧

### PDF Processing Pipeline

1. **Text Extraction**: Multiple extraction methods (PyPDF2, pdfplumber) with fallback support
2. **Text Cleaning**: Remove extra whitespace, normalize formatting
3. **Chunking Strategy**: Semantic chunking with configurable overlap
4. **Embedding Generation**: SentenceTransformers for high-quality embeddings

### Vector Database

* **Storage**: FAISS for efficient similarity search
* **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
* **Retrieval**: Top-k similarity search with cosine distance
* **Persistence**: Local storage for processed documents

### Mistral Integration

* **Model**: `mistral-large-latest` (configurable)
* **API**: Native Mistral Chat API via LangChain
* **Context Window**: Optimized chunk selection for context limits
* **Temperature**: Balanced for accuracy and creativity

## Configuration Options ⚙️

### Environment Variables

```bash
# Required
MISTRAL_API_KEY=your_api_key_here

# Optional (with defaults)
MISTRAL_MODEL=mistral-large-latest
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNKS=5
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Model Parameters

* **Chunk Size**: 500-2000 characters (default: 1000)
* **Overlap**: 100-500 characters (default: 200)
* **Top-K Retrieval**: 3-10 chunks (default: 5)
* **Temperature**: 0.0-1.0 (default: 0.3)

## Dependencies 📦

### Core Requirements

```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-mistralai>=0.1.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
PyPDF2>=3.0.1
pdfplumber>=0.9.0
python-dotenv>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
```

### Optional Dependencies

```txt
chromadb>=0.4.0          # Alternative vector database
mistralai>=0.1.0         # Direct Mistral API client
streamlit-chat>=0.1.0    # Enhanced chat UI components
```

## API Reference 📚

### PDFExtractor Class

```python
from pdf_utils import PDFExtractor

extractor = PDFExtractor()
text = extractor.extract_text(pdf_file)
```

**Methods:**
* `extract_text(file)`: Extract text from PDF file
* `clean_text(text)`: Clean and normalize extracted text

### VectorStore Class

```python
from vector_store import VectorStore

store = VectorStore()
store.create_index(text_chunks)
results = store.similarity_search(query, k=5)
```

**Methods:**
* `create_chunks(text)`: Split text into semantic chunks
* `create_index(chunks)`: Generate embeddings and build FAISS index
* `similarity_search(query, k)`: Retrieve top-k similar chunks

### QAChain Class

```python
from qa_chain import QAChain

qa = QAChain()
response = qa.answer_question(query, context_chunks)
```

**Methods:**
* `answer_question(query, chunks)`: Generate answer using Mistral
* `get_response_details()`: Get detailed response metadata

## Troubleshooting 🔧

### Common Issues

**1. PDF Extraction Fails**
```python
# Try different extraction methods
extractor = PDFExtractor(method='pdfplumber')  # or 'pypdf2'
```

**2. Empty Embeddings**
```python
# Check text chunking
chunks = store.create_chunks(text)
print(f"Created {len(chunks)} chunks")
```

**3. Mistral API Errors**
```python
# Verify API key and model availability
from config import MISTRAL_API_KEY, MISTRAL_MODEL
print(f"Using model: {MISTRAL_MODEL}")
```

### Performance Optimization

* **Large PDFs**: Increase chunk size for better context
* **Slow Queries**: Reduce top-k retrieval count
* **Memory Issues**: Use smaller embedding models
* **API Limits**: Implement request throttling

## Examples 💡

### Basic Usage

```python
# Upload and process PDF
uploaded_file = st.file_uploader("Choose PDF", type="pdf")
if uploaded_file:
    text = pdf_extractor.extract_text(uploaded_file)
    vector_store.create_index(text)
    
# Ask questions
query = st.text_input("Ask a question:")
if query:
    chunks = vector_store.similarity_search(query)
    answer = qa_chain.answer_question(query, chunks)
    st.write(answer)
```

### Advanced Features

```python
# Get document overview
overview = qa_chain.answer_question(
    "Provide a comprehensive overview of this document",
    vector_store.get_all_chunks()[:10]
)

# Extract key topics
topics = qa_chain.answer_question(
    "List the main topics and themes discussed",
    vector_store.similarity_search("main topics themes", k=8)
)
```

## Contributing 🤝

### Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Create `.env` file with your API keys
5. Run tests: `python -m pytest tests/`

### Code Style

* Follow PEP 8 guidelines
* Use type hints where possible
* Add docstrings for all functions
* Maintain modular architecture


## Changelog 📝

### v1.0.0 (Latest)
* Initial release with Mistral AI integration
* PDF processing with multiple extraction methods
* FAISS vector storage
* Interactive Streamlit interface
* Advanced response analysis features

## Support 💬

For questions, issues, or contributions:

* **GitHub Issues**: [Create an issue](https://github.com/Am3nek/chat-with-notes/issues)

* **Discussions**: [GitHub Discussions](https://github.com/Am3nek/chat-with-notes/discussions)

---

**Built with ❤️ using Streamlit, LangChain, and Mistral AI**
