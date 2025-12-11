from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
from pathlib import Path
from typing import List, Dict, Any

class DoclingParser:
    """
    Parses digital documents (PDF, DOCX, etc.) using Docling to preserve structure.
    """
    
    def __init__(self):
        # Use defaults which support PDF, DOCX, HTML, etc.
        self.converter = DocumentConverter()

    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a file and returns a structured dictionary.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        result: ConversionResult = self.converter.convert(path)
        
        # Extract text and structure
        document = result.document
        
        # Basic text extraction
        full_text = document.export_to_markdown()
        
        # Extract tables (simplified)
        tables = []
        # Check if tables exist before iterating
        if hasattr(document, 'tables') and document.tables:
            for table in document.tables:
                tables.append(table.export_to_dataframe().to_dict(orient="records"))

        return {
            "text": full_text,
            "tables": tables,
            "metadata": {
                "filename": path.name,
                "page_count": getattr(document, 'num_pages', 1),
                "origin": "docling"
            }
        }
