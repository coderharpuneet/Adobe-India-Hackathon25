import os
import json
from pathlib import Path
from ml_pdf_processor import process_pdfs

def process_pdfs():
    # Get input and output directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        # Process PDF using ML model
        processed_data = process_pdfs(pdf_file)
        
        # Create output JSON file
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w") as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"Processed {pdf_file.name} -> {output_file.name}")

if __name__ == "__main__":
    print("Starting ML-based PDF processing")
    process_pdfs()
    print("Completed ML-based PDF processing")
