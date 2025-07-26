#!/usr/bin/env python3
"""
Persona-Driven Document Intelligence System
Challenge 1B: Round 1B: Persona-Driven Document Intelligence

This system extracts and prioritizes relevant sections from document collections
based on specific personas and their job-to-be-done tasks.
"""

import os
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import argparse

# PDF processing libraries
import pdfplumber
import PyPDF2

# NLP and ML libraries
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


class PersonaDocumentAnalyzer:
    """
    Main class for analyzing documents based on persona and job-to-be-done.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # Reduced for memory efficiency
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            dtype=np.float32  # Use float32 for memory efficiency
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text content from PDF with page-wise information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        extracted_data = {
            'filename': os.path.basename(pdf_path),
            'pages': [],
            'full_text': '',
            'sections': []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        page_text = page_text.strip()
                        extracted_data['pages'].append({
                            'page_number': page_num,
                            'text': page_text,
                            'word_count': len(page_text.split())
                        })
                        extracted_data['full_text'] += f" {page_text}"
                
                # Extract sections based on text patterns
                extracted_data['sections'] = self._identify_sections(extracted_data['pages'])
                
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            
        return extracted_data
    
    def _identify_sections(self, pages: List[Dict]) -> List[Dict]:
        """
        Identify sections within the document based on text patterns.
        
        Args:
            pages: List of page dictionaries
            
        Returns:
            List of identified sections
        """
        sections = []
        
        for page in pages:
            text = page['text']
            lines = text.split('\n')
            
            current_section = None
            section_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line is a potential section header
                if self._is_section_header(line):
                    # Save previous section if exists
                    if current_section and section_content:
                        sections.append({
                            'title': current_section,
                            'content': ' '.join(section_content),
                            'page_number': page['page_number'],
                            'word_count': len(' '.join(section_content).split())
                        })
                    
                    # Start new section
                    current_section = line
                    section_content = []
                else:
                    if current_section:
                        section_content.append(line)
            
            # Add last section if exists
            if current_section and section_content:
                sections.append({
                    'title': current_section,
                    'content': ' '.join(section_content),
                    'page_number': page['page_number'],
                    'word_count': len(' '.join(section_content).split())
                })
        
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """
        Determine if a line is likely a section header.
        
        Args:
            line: Text line to check
            
        Returns:
            Boolean indicating if line is a section header
        """
        # Common patterns for section headers
        header_patterns = [
            r'^[A-Z][A-Za-z\s]+$',  # All caps or title case
            r'^Chapter \d+',          # Chapter numbers
            r'^\d+\.?\s+[A-Z]',      # Numbered sections
            r'^[IVX]+\.?\s+[A-Z]',   # Roman numerals
        ]
        
        # Check length (headers are usually shorter)
        if len(line.split()) > 10:
            return False
            
        # Check patterns
        for pattern in header_patterns:
            if re.match(pattern, line):
                return True
                
        # Check if line is in title case and shorter
        if line.istitle() and len(line.split()) <= 6:
            return True
            
        return False
    
    def create_persona_profile(self, persona: Dict, job_to_be_done: Dict) -> Dict:
        """
        Create a comprehensive persona profile combining role and task.
        
        Args:
            persona: Persona information
            job_to_be_done: Task information
            
        Returns:
            Combined persona profile
        """
        role = persona.get('role', '')
        task = job_to_be_done.get('task', '')
        
        # Create keyword sets based on persona and task
        persona_keywords = self._extract_keywords(role)
        task_keywords = self._extract_keywords(task)
        
        return {
            'role': role,
            'task': task,
            'persona_keywords': persona_keywords,
            'task_keywords': task_keywords,
            'combined_text': f"{role} {task}",
            'focus_areas': self._identify_focus_areas(role, task)
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        words = word_tokenize(text.lower())
        keywords = [word for word in words if word.isalpha() and word not in self.stop_words]
        return keywords
    
    def _identify_focus_areas(self, role: str, task: str) -> List[str]:
        """
        Identify focus areas based on role and task.
        
        Args:
            role: Persona role
            task: Job to be done
            
        Returns:
            List of focus areas
        """
        focus_mapping = {
            'travel planner': ['itinerary', 'destinations', 'accommodations', 'activities', 'budget', 'transportation'],
            'hr professional': ['forms', 'compliance', 'onboarding', 'documentation', 'procedures', 'regulations'],
            'food contractor': ['menu', 'recipes', 'ingredients', 'dietary', 'preparation', 'catering'],
            'researcher': ['methodology', 'data', 'analysis', 'literature', 'findings', 'experiments'],
            'student': ['concepts', 'theory', 'examples', 'practice', 'preparation', 'fundamentals'],
            'analyst': ['trends', 'metrics', 'performance', 'comparison', 'insights', 'data']
        }
        
        role_lower = role.lower()
        task_lower = task.lower()
        
        # Find matching focus areas
        focus_areas = []
        for key, areas in focus_mapping.items():
            if key in role_lower or any(word in task_lower for word in key.split()):
                focus_areas.extend(areas)
        
        # Add task-specific keywords
        task_keywords = self._extract_keywords(task)
        focus_areas.extend(task_keywords[:5])  # Top 5 task keywords
        
        return list(set(focus_areas))  # Remove duplicates
    
    def analyze_document_relevance(self, documents: List[Dict], persona_profile: Dict) -> List[Dict]:
        """
        Analyze relevance of documents and sections to persona and task.
        
        Args:
            documents: List of document data
            persona_profile: Persona profile dictionary
            
        Returns:
            List of documents with relevance scores
        """
        all_sections = []
        doc_section_mapping = []
        
        # Collect all sections from all documents
        for doc in documents:
            for section in doc['sections']:
                all_sections.append(section['content'])
                doc_section_mapping.append({
                    'document': doc['filename'],
                    'section': section,
                    'doc_index': documents.index(doc)
                })
        
        if not all_sections:
            return documents
        
        # Create TF-IDF vectors for sections
        section_vectors = self.vectorizer.fit_transform(all_sections)
        
        # Create persona vector
        persona_text = persona_profile['combined_text'] + ' ' + ' '.join(persona_profile['focus_areas'])
        persona_vector = self.vectorizer.transform([persona_text])
        
        # Calculate relevance scores
        relevance_scores = cosine_similarity(section_vectors, persona_vector).flatten()
        
        # Add relevance scores to sections
        for i, mapping in enumerate(doc_section_mapping):
            mapping['section']['relevance_score'] = float(relevance_scores[i])
        
        # Update documents with scored sections
        for doc in documents:
            doc['sections'] = [s for s in doc['sections'] if s.get('relevance_score', 0) > 0.1]
            doc['sections'].sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return documents
    
    def extract_top_sections(self, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Extract top-k most relevant sections across all documents.
        
        Args:
            documents: List of analyzed documents
            top_k: Number of top sections to extract
            
        Returns:
            List of top sections with metadata
        """
        all_sections = []
        
        for doc in documents:
            for section in doc['sections']:
                all_sections.append({
                    'document': doc['filename'],
                    'section_title': section['title'],
                    'content': section['content'],
                    'page_number': section['page_number'],
                    'relevance_score': section.get('relevance_score', 0),
                    'word_count': section.get('word_count', 0)
                })
        
        # Sort by relevance score and return top-k
        all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Add importance rank
        for i, section in enumerate(all_sections[:top_k]):
            section['importance_rank'] = i + 1
        
        return all_sections[:top_k]
    
    def generate_subsection_analysis(self, top_sections: List[Dict], max_length: int = 500) -> List[Dict]:
        """
        Generate refined text analysis for top sections.
        
        Args:
            top_sections: List of top relevant sections
            max_length: Maximum length for refined text
            
        Returns:
            List of subsection analyses
        """
        subsection_analyses = []
        
        for section in top_sections:
            content = section['content']
            
            # Extract key sentences
            sentences = sent_tokenize(content)
            if len(sentences) <= 3:
                refined_text = content
            else:
                # Score sentences based on keyword presence
                sentence_scores = []
                for sentence in sentences:
                    score = len([word for word in sentence.lower().split() if word not in self.stop_words])
                    sentence_scores.append((sentence, score))
                
                # Sort by score and take top sentences
                sentence_scores.sort(key=lambda x: x[1], reverse=True)
                top_sentences = [s[0] for s in sentence_scores[:3]]
                refined_text = ' '.join(top_sentences)
            
            # Truncate if too long
            if len(refined_text) > max_length:
                refined_text = refined_text[:max_length] + "..."
            
            subsection_analyses.append({
                'document': section['document'],
                'refined_text': refined_text,
                'page_number': section['page_number']
            })
        
        return subsection_analyses
    
    def process_document_collection(self, input_data: Dict, pdf_directory: str) -> Dict:
        """
        Main processing function for a document collection.
        
        Args:
            input_data: Input JSON data
            pdf_directory: Directory containing PDF files
            
        Returns:
            Processed output data
        """
        start_time = time.time()
        
        # Extract documents
        documents = []
        for doc_info in input_data['documents']:
            pdf_path = os.path.join(pdf_directory, doc_info['filename'])
            if os.path.exists(pdf_path):
                print(f"Processing: {doc_info['filename']}")
                doc_data = self.extract_text_from_pdf(pdf_path)
                doc_data['title'] = doc_info.get('title', doc_info['filename'])
                documents.append(doc_data)
            else:
                print(f"Warning: File not found: {pdf_path}")
        
        # Create persona profile
        persona_profile = self.create_persona_profile(
            input_data['persona'],
            input_data['job_to_be_done']
        )
        
        # Analyze document relevance
        analyzed_documents = self.analyze_document_relevance(documents, persona_profile)
        
        # Extract top sections
        top_sections = self.extract_top_sections(analyzed_documents, top_k=5)
        
        # Generate subsection analysis
        subsection_analysis = self.generate_subsection_analysis(top_sections)
        
        # Prepare output
        output_data = {
            'metadata': {
                'input_documents': [doc['filename'] for doc in input_data['documents']],
                'persona': input_data['persona']['role'],
                'job_to_be_done': input_data['job_to_be_done']['task'],
                'processing_timestamp': datetime.now().isoformat()
            },
            'extracted_sections': [
                {
                    'document': section['document'],
                    'section_title': section['section_title'],
                    'importance_rank': section['importance_rank'],
                    'page_number': section['page_number']
                }
                for section in top_sections
            ],
            'subsection_analysis': subsection_analysis
        }
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        return output_data


def main():
    """Main function to run the persona document analyzer."""
    parser = argparse.ArgumentParser(description='Persona-Driven Document Intelligence System')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--pdf_dir', required=True, help='Directory containing PDF files')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Load input data
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    # Initialize analyzer
    analyzer = PersonaDocumentAnalyzer()
    
    # Process documents
    output_data = analyzer.process_document_collection(input_data, args.pdf_dir)
    
    # Save output
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Analysis complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
