"""
PDF text extraction utilities.
"""
import PyPDF2
import pdfplumber
import re
from io import BytesIO
from typing import Optional


class PDFExtractor:
    """Handles PDF text extraction with multiple fallback methods."""
    
    def __init__(self):
        pass
    
    def extract_text_pypdf2(self, pdf_file: BytesIO) -> str:
        """Extract text using PyPDF2."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_file: BytesIO) -> str:
        """Extract text using pdfplumber (better for complex PDFs)."""
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix common OCR issues
        text = text.replace('—', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_and_clean(self, pdf_file: BytesIO) -> Optional[str]:
        """
        Extract and clean text from PDF using the best available method.
        
        Args:
            pdf_file: BytesIO object containing PDF data
            
        Returns:
            Cleaned text string or None if extraction fails
        """
        # Reset file pointer
        pdf_file.seek(0)
        
        # Try pdfplumber first (better for complex layouts)
        text = self.extract_text_pdfplumber(pdf_file)
        
        # Fallback to PyPDF2 if pdfplumber fails or returns empty
        if not text or len(text.strip()) < 100:
            pdf_file.seek(0)
            text = self.extract_text_pypdf2(pdf_file)
        
        # Clean the extracted text
        if text:
            text = self.clean_text(text)
            
            # Check if we have meaningful content
            if len(text.strip()) < 50:
                return None
                
            return text
        
        return None
    
    def get_pdf_metadata(self, pdf_file: BytesIO) -> dict:
        """Extract basic metadata from PDF."""
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'title': pdf_reader.metadata.get('/Title', 'Unknown') if pdf_reader.metadata else 'Unknown',
                'author': pdf_reader.metadata.get('/Author', 'Unknown') if pdf_reader.metadata else 'Unknown',
                'subject': pdf_reader.metadata.get('/Subject', 'Unknown') if pdf_reader.metadata else 'Unknown'
            }
            
            return metadata
        except Exception as e:
            print(f"Metadata extraction failed: {e}")
            return {'num_pages': 0, 'title': 'Unknown', 'author': 'Unknown', 'subject': 'Unknown'}