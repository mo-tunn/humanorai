from fastapi import UploadFile, HTTPException
import io
import docx
import pypdf

def extract_text_from_file(file: UploadFile) -> str:
    filename = file.filename.lower()
    text = ""

    try:
        content = file.file.read()
        if filename.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif filename.endswith('.docx') or filename.endswith('.doc'):
            try:
                doc = docx.Document(io.BytesIO(content))
                text = "\n".join([para.text for para in doc.paragraphs])
            except Exception:
                 raise HTTPException(status_code=400, detail="Old .doc format might not be supported. Please convert to .docx or .pdf")
        elif filename.endswith('.txt'):
            text = content.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
            
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        # If it's already an HTTPException, re-raise it
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
    return text
