#!/usr/bin/env python3
"""
Script to evaluate the trained model against sample outputs
"""

import sys
import os
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pdf_processor import PDFOutlineExtractor

def calculate_accuracy(predicted, actual):
    """Calculate accuracy metrics"""
    # Title accuracy
    title_match = predicted['title'].strip().lower() == actual['title'].strip().lower()
    
    # Outline accuracy
    pred_outline_texts = set(item['text'].lower() for item in predicted['outline'])
    actual_outline_texts = set(item['text'].lower() for item in actual['outline'])
    
    if len(actual_outline_texts) == 0:
        outline_precision = 1.0 if len(pred_outline_texts) == 0 else 0.0
        outline_recall = 1.0
    else:
        intersection = pred_outline_texts.intersection(actual_outline_texts)
        outline_precision = len(intersection) / len(pred_outline_texts) if pred_outline_texts else 0.0
        outline_recall = len(intersection) / len(actual_outline_texts)
    
    outline_f1 = 2 * (outline_precision * outline_recall) / (outline_precision + outline_recall) if (outline_precision + outline_recall) > 0 else 0.0
    
    return {
        'title_match': title_match,
        'outline_precision': outline_precision,
        'outline_recall': outline_recall,
        'outline_f1': outline_f1
    }

def main():
    print("Evaluating PDF Outline Extraction Model...")
    
    extractor = PDFOutlineExtractor()
    
    # Load model
    model_path = "pdf_outline_model.pkl"
    if not os.path.exists(model_path):
        print("Model not found! Please train the model first.")
        return
    
    extractor.load_model(model_path)
    
    # Evaluate on sample files
    sample_dir = Path("sample_dataset/outputs")
    pdf_dir = Path("sample_dataset/pdfs")
    
    results = []
    
    for sample_file in sample_dir.glob("*.json"):
        pdf_file = pdf_dir / f"{sample_file.stem}.pdf"
        
        if not pdf_file.exists():
            continue
        
        # Load actual output
        with open(sample_file, 'r') as f:
            actual = json.load(f)
        
        # Predict
        try:
            predicted = extractor.predict_structure(str(pdf_file))
            
            # Calculate metrics
            metrics = calculate_accuracy(predicted, actual)
            metrics['file'] = sample_file.stem
            results.append(metrics)
            
            print(f"\n{sample_file.stem}:")
            print(f"  Title Match: {metrics['title_match']}")
            print(f"  Outline F1: {metrics['outline_f1']:.3f}")
            print(f"  Predicted Title: {predicted['title'][:50]}...")
            print(f"  Actual Title: {actual['title'][:50]}...")
            
        except Exception as e:
            print(f"Error processing {sample_file.stem}: {e}")
    
    # Overall metrics
    if results:
        avg_title_accuracy = sum(r['title_match'] for r in results) / len(results)
        avg_outline_f1 = sum(r['outline_f1'] for r in results) / len(results)
        
        print(f"\n=== Overall Results ===")
        print(f"Average Title Accuracy: {avg_title_accuracy:.3f}")
        print(f"Average Outline F1: {avg_outline_f1:.3f}")
        print(f"Files processed: {len(results)}")

if __name__ == "__main__":
    main()
