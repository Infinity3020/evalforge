# EvalForge 🔬

> Open source LLM evaluation framework. Know if your AI is actually working.

**Live demo:** https://evalforge-git-453855922097.europe-west1.run.app

---

## What is EvalForge?

EvalForge automatically scores your LLM outputs by comparing AI answers against expected answers. It catches bad answers before your users do.

Built with BERT embeddings, FastAPI, and support for multiple AI judges including Claude and GPT-4.

---

## Scoring Methods

| Method | Cost | How it works |
|--------|------|--------------|
| Local BERT | Free | Converts sentences to vectors, measures semantic similarity |
| Claude | Your API key | Claude reads both answers and scores them 0-10 |
| GPT-4 | Your API key | GPT-4 reads both answers and scores them 0-10 |
| Local + Claude | Your API key | Runs both, averages the scores |
| Local + GPT-4 | Your API key | Runs both, averages the scores |
| All three | Your API key | Ensemble of all 3 methods |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Infinity3020/evalforge.git
cd evalforge

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn sentence-transformers scikit-learn numpy python-dotenv aiofiles openai anthropic

# Run the server
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000` to see the website.

---

## API Usage

```bash
curl -X POST https://evalforge-git-453855922097.europe-west1.run.app/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["What is your return policy?"],
    "expected_answers": ["30 day return with receipt"],
    "predicted_answers": ["Returns accepted within one month"],
    "method": "local"
  }'
```

Response:
```json
{
  "overall_score": 8.5,
  "total_questions": 1,
  "passed": 1,
  "failed": 0,
  "pass_rate": "100.0%",
  "results": [...]
}
```

---

## Tech Stack

- **FastAPI** — REST API framework
- **Sentence Transformers** — BERT embeddings for semantic similarity
- **scikit-learn** — cosine similarity scoring
- **OpenAI / Anthropic** — LLM-as-judge scoring
- **Docker** — containerized deployment
- **Google Cloud Run** — production hosting

---

## License

MIT — free to use for any purpose.