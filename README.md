# EvalForge 🔍

**Open source LLM evaluation framework. Know if your AI is actually working.**

EvalForge automatically scores your LLM outputs using BERT semantic similarity. Catch bad answers before your users do.

## Live Demo
[Try EvalForge](https://evalforge.onrender.com) ← we will add this after deployment

## What it does
- Scores AI answers against expected answers using BERT embeddings
- Measures semantic similarity, not just word matching
- Shows overall score, pass rate, and per-question breakdown
- Supports local scoring (free), Claude, and GPT-4 as judges
- REST API so you can plug it into your CI pipeline

## Quick Start

```bash
git clone https://github.com/Infinity3020/evalforge.git
cd evalforge
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn sentence-transformers pandas openai python-dotenv aiofiles
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 in your browser.

## API Usage

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["What is your return policy?"],
    "expected_answers": ["30 day return with receipt"],
    "predicted_answers": ["Returns accepted within one month"]
  }'
```

## Tech Stack
- FastAPI — REST API framework
- Sentence Transformers — BERT semantic similarity scoring
- Python — core language
- Uvicorn — ASGI server

## License
MIT — free to use for any purpose