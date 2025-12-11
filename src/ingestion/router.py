import os
from typing import Dict, Any, List
from .docling_parser import DoclingParser
from .metadata import MetadataExtractor
from .chunking import Chunker

# OCR is optional - disabled by default due to high resource requirements
ENABLE_OCR = os.getenv("ENABLE_OCR", "false").lower() == "true"

if ENABLE_OCR:
    try:
        from .deepseek_ocr import DeepSeekOCR
        OCR_AVAILABLE = True
    except ImportError:
        print("Warning: OCR enabled but deepseek_ocr module not found")
        OCR_AVAILABLE = False
else:
    OCR_AVAILABLE = False

class IngestionPipeline:
    """
    Main ETL Pipeline:
    1. Route to Parser (Docling vs DeepSeek OCR)
    2. Extract Metadata
    3. Chunk Content
    
    OCR is disabled by default. Enable with ENABLE_OCR=true environment variable.
    Note: OCR requires significant GPU resources.
    """
    
    def __init__(self):
        self.docling = DoclingParser()
        self.ocr = DeepSeekOCR() if (ENABLE_OCR and OCR_AVAILABLE) else None
        self.metadata_extractor = MetadataExtractor()
        self.chunker = Chunker(strategy="hierarchical")
        
        if ENABLE_OCR and not OCR_AVAILABLE:
            print("Warning: OCR requested but not available")
        elif ENABLE_OCR:
            print("OCR enabled - image files will be processed")
        else:
            print("OCR disabled - only digital documents supported")

    def process_document(self, file_path: str) -> List[Any]:
        """
        Full processing flow for a single document.
        """
        print(f"Processing: {file_path}")
        
        # 1. Ingestion Router
        file_ext = os.path.splitext(file_path)[1].lower()
        
        content = {}
        if file_ext in ['.pdf', '.docx', '.html', '.md']:
            # Use Docling for digital docs
            try:
                print("Routing to Docling...")
                content = self.docling.parse(file_path)
            except Exception as e:
                print(f"Docling failed: {e}. Falling back to OCR if available.")
                if self.ocr:
                    content = self.ocr.process_image(file_path)
                else:
                    return []
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff'] and self.ocr:
            print("Routing to DeepSeek-OCR...")
            content = self.ocr.process_image(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff']:
            print(f"Image file detected but OCR is disabled. Enable with ENABLE_OCR=true")
            return []
        else:
            print(f"Unsupported file type: {file_ext}. Supported: .pdf, .docx, .html, .md" + 
                  (" and images with OCR" if self.ocr else ""))
            return []

        if not content or not content.get("text"):
            print("No text extracted.")
            return []

        # 2. Metadata Enrichment
        print("Extracting Metadata...")
        enriched_metadata = self.metadata_extractor.extract(content["text"], os.path.basename(file_path))
        
        # Merge technical metadata (from parser) with semantic metadata (from LLM)
        final_metadata = {**content.get("metadata", {}), **enriched_metadata}
        
        # 3. Chunking
        print("Chunking Document...")
        nodes = self.chunker.chunk(content["text"], final_metadata)
        
        print(f"Generated {len(nodes)} chunks.")
        return nodes

if __name__ == "__main__":
    # Test run
    pipeline = IngestionPipeline()
    # pipeline.process_document("example.pdf")

