import pytest
from unittest.mock import MagicMock, patch
from fastapi import UploadFile, HTTPException
import io
import sys
import os

# Add backend to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_parser import extract_text_from_file

def create_upload_file(filename, content):
    return UploadFile(filename=filename, file=io.BytesIO(content))

def test_extract_txt():
    """Test extracting text from .txt file."""
    content = b"Hello world"
    file = create_upload_file("test.txt", content)
    result = extract_text_from_file(file)
    assert result == "Hello world"

@patch("utils.file_parser.pypdf.PdfReader")
def test_extract_pdf(mock_pdf_reader):
    """Test extracting text from .pdf file."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF Page Content"
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page, mock_page]
    mock_pdf_reader.return_value = mock_pdf
    
    file = create_upload_file("test.pdf", b"%PDF-dummy")
    result = extract_text_from_file(file)
    
    assert "PDF Page Content" in result
    # Should appear twice due to 2 pages
    assert result.count("PDF Page Content") == 2

@patch("utils.file_parser.docx.Document")
def test_extract_docx(mock_document):
    """Test extracting text from .docx file."""
    mock_para = MagicMock()
    mock_para.text = "Docx Para Content"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para]
    mock_document.return_value = mock_doc
    
    file = create_upload_file("test.docx", b"PK-dummy")
    result = extract_text_from_file(file)
    
    assert "Docx Para Content" in result

def test_extract_unsupported():
    """Test unsupported file extension."""
    file = create_upload_file("test.exe", b"binary")
    with pytest.raises(HTTPException) as excinfo:
        extract_text_from_file(file)
    assert excinfo.value.status_code == 400
    assert "Unsupported file type" in excinfo.value.detail

@patch("utils.file_parser.docx.Document")
def test_extract_docx_error(mock_document):
    """Test error handling when docx parsing fails."""
    mock_document.side_effect = Exception("Corrupt file")
    
    file = create_upload_file("bad.docx", b"bad")
    with pytest.raises(HTTPException) as excinfo:
        extract_text_from_file(file)
    assert excinfo.value.status_code == 400
    assert "Old .doc format" in excinfo.value.detail

def test_extract_reader_crash():
    """Test generic crash in reader loop."""
    # Testing the outer try/except block
    # We can pass an object that crashes on .file.read() or similar
    
    mock_file = MagicMock()
    mock_file.filename = "crash.txt"
    mock_file.file.read.side_effect = Exception("Disk error")
    
    with pytest.raises(HTTPException) as excinfo:
        extract_text_from_file(mock_file)
    
    assert excinfo.value.status_code == 400
    assert "Error reading file" in excinfo.value.detail


