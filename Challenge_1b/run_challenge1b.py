#!/usr/bin/env python3
"""
Production execution script for Challenge 1B.
This script provides robust execution with proper error handling and logging.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('challenge1b_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

try:
    from persona_document_analyzer import PersonaDocumentAnalyzer
except ImportError as e:
    logger.error(f"Failed to import PersonaDocumentAnalyzer: {e}")
    sys.exit(1)


def validate_environment():
    """Validate that all required dependencies are available."""
    logger.info("Validating environment...")
    
    required_modules = [
        'pdfplumber', 'PyPDF2', 'nltk', 'sklearn', 'numpy'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False
    
    logger.info("✅ Environment validation passed")
    return True


def process_single_collection(collection_path: str, output_suffix: str = "_generated") -> bool:
    """
    Process a single collection with comprehensive error handling.
    
    Args:
        collection_path: Path to the collection directory
        output_suffix: Suffix to add to output filename
        
    Returns:
        Success status
    """
    collection_name = os.path.basename(collection_path)
    logger.info(f"Processing {collection_name}...")
    
    start_time = time.time()
    
    try:
        # File paths
        input_file = os.path.join(collection_path, "challenge1b_input.json")
        pdf_dir = os.path.join(collection_path, "PDFs")
        output_file = os.path.join(collection_path, f"challenge1b_output{output_suffix}.json")
        
        # Validate input files
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False
        
        if not os.path.exists(pdf_dir):
            logger.error(f"PDF directory not found: {pdf_dir}")
            return False
        
        # Load and validate input data
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        required_keys = ['documents', 'persona', 'job_to_be_done']
        for key in required_keys:
            if key not in input_data:
                logger.error(f"Missing required key in input: {key}")
                return False
        
        # Check PDF file availability
        available_docs = []
        missing_docs = []
        
        for doc_info in input_data['documents']:
            pdf_path = os.path.join(pdf_dir, doc_info['filename'])
            if os.path.exists(pdf_path):
                available_docs.append(doc_info)
            else:
                missing_docs.append(doc_info['filename'])
        
        if missing_docs:
            logger.warning(f"Missing PDF files: {missing_docs}")
        
        if not available_docs:
            logger.error("No PDF files available for processing")
            return False
        
        # Update input data with available documents only
        input_data['documents'] = available_docs
        
        logger.info(f"Processing {len(available_docs)} documents...")
        
        # Initialize analyzer
        analyzer = PersonaDocumentAnalyzer()
        
        # Process documents
        output_data = analyzer.process_document_collection(input_data, pdf_dir)
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        processing_time = time.time() - start_time
        
        # Log results
        logger.info(f"✅ {collection_name} processed successfully")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        logger.info(f"   Documents processed: {len(available_docs)}")
        logger.info(f"   Sections extracted: {len(output_data['extracted_sections'])}")
        logger.info(f"   Output saved to: {output_file}")
        
        # Validate processing time constraint
        if processing_time > 60:
            logger.warning(f"Processing time exceeded 60 seconds: {processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {collection_name}: {str(e)}")
        logger.error(f"Processing time: {time.time() - start_time:.2f} seconds")
        return False


def main():
    """Main execution function."""
    logger.info("🚀 Starting Challenge 1B: Persona-Driven Document Intelligence")
    logger.info(f"Execution time: {datetime.now().isoformat()}")
    
    # Validate environment
    if not validate_environment():
        return 1
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Working directory: {script_dir}")
    
    # Find collections
    collections = []
    for item in os.listdir(script_dir):
        item_path = os.path.join(script_dir, item)
        if os.path.isdir(item_path) and item.startswith("Collection "):
            collections.append(item_path)
    
    collections.sort()  # Process in order
    
    if not collections:
        logger.error("No collections found in the current directory")
        return 1
    
    logger.info(f"Found {len(collections)} collections to process")
    
    # Process each collection
    success_count = 0
    total_start_time = time.time()
    
    for collection_path in collections:
        if process_single_collection(collection_path):
            success_count += 1
        else:
            logger.error(f"Failed to process: {os.path.basename(collection_path)}")
    
    total_time = time.time() - total_start_time
    
    # Summary
    logger.info(f"\n📊 Execution Summary")
    logger.info(f"✅ Successfully processed: {success_count}/{len(collections)} collections")
    logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
    logger.info(f"📝 Log file: challenge1b_execution.log")
    
    if success_count == len(collections):
        logger.info("🎉 All collections processed successfully!")
        return 0
    else:
        logger.error("⚠️  Some collections failed to process")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
