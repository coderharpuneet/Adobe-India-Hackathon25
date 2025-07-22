#!/usr/bin/env python3
"""
Script to train the PDF outline extraction model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pdf_processor import PDFOutlineExtractor

def main():
    print("Training PDF Outline Extraction Model...")
    
    extractor = PDFOutlineExtractor()
    
    # Train on sample data
    extractor.train("sample_dataset/outputs")
    
    # Save the model
    extractor.save_model("pdf_outline_model.pkl")
    
    print("Model training completed!")
    
    # Test on a sample file
    print("\nTesting on sample file...")
    try:
        result = extractor.predict_structure("sample_dataset/pdfs/file03.pdf")
        print(f"Title: {result['title']}")
        print(f"Outline items: {len(result['outline'])}")
        for item in result['outline'][:5]:  # Show first 5 items
            print(f"  {item['level']}: {item['text']} (Page {item['page']})")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
