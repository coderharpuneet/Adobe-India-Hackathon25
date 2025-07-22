import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pdfplumber
import PyPDF2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class PDFOutlineExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def extract_text_with_metadata(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page numbers and formatting info"""
        text_blocks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text with bounding boxes
                    chars = page.chars
                    if not chars:
                        continue
                        
                    # Group characters into lines
                    lines = self._group_chars_into_lines(chars)
                    
                    for line in lines:
                        if line['text'].strip():
                            text_blocks.append({
                                'text': line['text'].strip(),
                                'page': page_num + 1,
                                'font_size': line.get('size', 12),
                                'is_bold': line.get('bold', False),
                                'y_position': line.get('y', 0)
                            })
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            # Fallback to PyPDF2
            return self._extract_with_pypdf2(pdf_path)
            
        return text_blocks
    
    def _group_chars_into_lines(self, chars: List[Dict]) -> List[Dict]:
        """Group characters into lines based on y-position"""
        if not chars:
            return []
            
        # Sort by y-position (top to bottom) then x-position (left to right)
        chars = sorted(chars, key=lambda c: (-c['y0'], c['x0']))
        
        lines = []
        current_line = []
        current_y = None
        tolerance = 2  # pixels
        
        for char in chars:
            if current_y is None or abs(char['y0'] - current_y) <= tolerance:
                current_line.append(char)
                current_y = char['y0']
            else:
                if current_line:
                    line_text = ''.join([c['text'] for c in current_line])
                    line_info = {
                        'text': line_text,
                        'size': current_line[0].get('size', 12),
                        'bold': any(c.get('fontname', '').lower().find('bold') != -1 for c in current_line),
                        'y': current_y
                    }
                    lines.append(line_info)
                
                current_line = [char]
                current_y = char['y0']
        
        # Add the last line
        if current_line:
            line_text = ''.join([c['text'] for c in current_line])
            line_info = {
                'text': line_text,
                'size': current_line[0].get('size', 12),
                'bold': any(c.get('fontname', '').lower().find('bold') != -1 for c in current_line),
                'y': current_y
            }
            lines.append(line_info)
        
        return lines
    
    def _extract_with_pypdf2(self, pdf_path: str) -> List[Dict]:
        """Fallback extraction using PyPDF2"""
        text_blocks = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    lines = text.split('\n')
                    for line in lines:
                        if line.strip():
                            text_blocks.append({
                                'text': line.strip(),
                                'page': page_num + 1,
                                'font_size': 12,  # Default
                                'is_bold': False,  # Default
                                'y_position': 0
                            })
        except Exception as e:
            print(f"Fallback extraction failed for {pdf_path}: {e}")
        
        return text_blocks
    
    def extract_features(self, text_block: Dict) -> List[float]:
        """Extract features for ML classification"""
        text = text_block['text']
        
        features = [
            # Text length features
            len(text),
            len(text.split()),
            
            # Font features
            text_block.get('font_size', 12),
            int(text_block.get('is_bold', False)),
            
            # Position features
            text_block.get('y_position', 0),
            
            # Text pattern features
            int(bool(re.match(r'^\d+\.', text))),  # Starts with number
            int(bool(re.match(r'^[A-Z][A-Z\s]+$', text))),  # All caps
            int(text.isupper()),  # Is uppercase
            int(text.istitle()),  # Is title case
            
            # Punctuation features
            text.count('.'),
            text.count(','),
            text.count(':'),
            text.count(';'),
            
            # Word features
            len([w for w in text.split() if w.isupper()]),  # Uppercase words
            len([w for w in text.split() if len(w) > 6]),   # Long words
        ]
        
        return features
    
    def prepare_training_data(self, sample_outputs_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from sample outputs"""
        X = []
        y = []
        
        # Load sample outputs
        sample_files = Path(sample_outputs_dir).glob('*.json')
        
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                sample_data = json.load(f)
            
            # Find corresponding PDF
            pdf_name = sample_file.stem + '.pdf'
            pdf_path = Path('sample_dataset/pdfs') / pdf_name
            
            if not pdf_path.exists():
                continue
            
            # Extract text blocks from PDF
            text_blocks = self.extract_text_with_metadata(str(pdf_path))
            
            # Create training examples
            outline_texts = {item['text']: item['level'] for item in sample_data.get('outline', [])}
            title_text = sample_data.get('title', '').strip()
            
            for block in text_blocks:
                features = self.extract_features(block)
                text = block['text']
                
                # Determine label
                if text == title_text:
                    label = 'TITLE'
                elif text in outline_texts:
                    label = outline_texts[text]
                else:
                    # Check for partial matches
                    label = 'BODY'
                    for outline_text, level in outline_texts.items():
                        if self._text_similarity(text, outline_text) > 0.8:
                            label = level
                            break
                
                X.append(features)
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def train(self, sample_outputs_dir: str):
        """Train the ML model"""
        print("Preparing training data...")
        X, y = self.prepare_training_data(sample_outputs_dir)
        
        if len(X) == 0:
            print("No training data found!")
            return
        
        print(f"Training on {len(X)} samples...")
        print(f"Labels: {set(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        print("Training completed!")
    
    def predict_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Predict document structure for a PDF"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        text_blocks = self.extract_text_with_metadata(pdf_path)
        
        if not text_blocks:
            return {"title": "", "outline": []}
        
        # Extract features and predict
        features = [self.extract_features(block) for block in text_blocks]
        predictions = self.classifier.predict(features)
        
        # Process predictions
        title = ""
        outline = []
        
        for block, pred in zip(text_blocks, predictions):
            if pred == 'TITLE' and not title:
                title = block['text']
            elif pred in ['H1', 'H2', 'H3', 'H4']:
                outline.append({
                    'level': pred,
                    'text': block['text'],
                    'page': block['page']
                })
        
        # If no title found, try to extract from first few blocks
        if not title and text_blocks:
            # Look for the longest text in first page that's not an outline item
            first_page_blocks = [b for b in text_blocks if b['page'] == 1]
            if first_page_blocks:
                title_candidate = max(first_page_blocks, key=lambda x: len(x['text']))
                if title_candidate['text'] not in [item['text'] for item in outline]:
                    title = title_candidate['text']
        
        return {
            "title": title,
            "outline": outline
        }
    
    def save_model(self, model_path: str):
        """Save trained model"""
        model_data = {
            'classifier': self.classifier,
            'is_trained': self.is_trained
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {model_path}")

def process_pdfs():
    """Main processing function"""
    print("Starting ML-based PDF processing...")
    
    # Initialize extractor
    extractor = PDFOutlineExtractor()
    
    # Check if model exists
    model_path = "pdf_outline_model.pkl"
    if os.path.exists(model_path):
        print("Loading existing model...")
        extractor.load_model(model_path)
    else:
        print("Training new model...")
        extractor.train("sample_dataset/outputs")
        extractor.save_model(model_path)
    
    # Process PDFs
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # For local testing, use sample dataset
    if not input_dir.exists():
        input_dir = Path("sample_dataset/pdfs")
        output_dir = Path("ml_outputs")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing {pdf_file.name}...")
            result = extractor.predict_structure(str(pdf_file))
            
            # Save result
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"Processed {pdf_file.name} -> {output_file.name}")
            print(f"  Title: {result['title'][:50]}...")
            print(f"  Outline items: {len(result['outline'])}")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    process_pdfs()
