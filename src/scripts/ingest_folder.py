import os
import asyncio
from src.ingestion.router import IngestionPipeline
from src.retrieval.qdrant_client import QdrantRetriever

async def ingest_folder(folder_path: str = "data"):
    """
    Scans the 'data/' folder and ingests all files into the RAG system.
    """
    print(f"Scanning folder: {folder_path}...")
    
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist. Creating it...")
        os.makedirs(folder_path)
        print("Please put your documents in the 'data/' folder and run this script again.")
        return

    # Initialize Pipeline
    pipeline = IngestionPipeline()
    retriever = QdrantRetriever()

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if not files:
        print("No files found in 'data/'.")
        return

    print(f"Found {len(files)} files. Starting ingestion...")

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            print(f"\n--- Processing: {filename} ---")
            nodes = pipeline.process_document(file_path)
            
            if nodes:
                retriever.upsert_nodes(nodes)
                print(f"Successfully ingested {filename}")
            else:
                print(f"Skipped {filename} (No content extracted)")
                
        except Exception as e:
            print(f"Failed to ingest {filename}: {e}")

    print("\nAll files processed!")

if __name__ == "__main__":
    asyncio.run(ingest_folder())
