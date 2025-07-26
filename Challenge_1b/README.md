# Challenge 1B: Persona-Driven Document Intelligence

## Quick Start

### Docker (Recommended)
```bash
docker build -t persona-doc-analyzer .
docker run -v "$(pwd):/app" persona-doc-analyzer
```

### Direct Python
```bash
pip install -r requirements.txt
python run_challenge1b.py
```

### Individual Collection
```bash
python persona_document_analyzer.py \
    --input "Collection 1/challenge1b_input.json" \
    --pdf_dir "Collection 1/PDFs" \
    --output "Collection 1/results.json"
```

## Files

- `persona_document_analyzer.py` - Main analysis engine
- `run_challenge1b.py` - Production runner for all collections
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `approach_explanation.md` - Methodology (300-500 words)
- `SOLUTION_OVERVIEW.md` - Complete documentation
- `Collection 1/2/3/` - Test datasets

## Requirements Met

✅ CPU Only | ✅ <1GB Model | ✅ <60s Processing | ✅ No Internet | ✅ Generic Solution
