from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from app.scorer import evaluate_dataset

app = FastAPI(
    title="EvalForge",
    description="Open source LLM evaluation framework",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class EvaluationRequest(BaseModel):
    questions: List[str]
    expected_answers: List[str]
    predicted_answers: List[str]

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    results = evaluate_dataset(
        questions=request.questions,
        expected_answers=request.expected_answers,
        predicted_answers=request.predicted_answers
    )
    return results

@app.get("/demo")
def demo():
    questions = [
        "What is your return policy?",
        "How long does shipping take?",
        "Do you offer free shipping?",
        "What payment methods do you accept?",
        "How do I track my order?"
    ]
    expected_answers = [
        "You can return items within 30 days with a receipt",
        "Standard shipping takes 5 to 7 business days",
        "We offer free shipping on orders over $50",
        "We accept Visa, Mastercard, PayPal and Apple Pay",
        "You will receive a tracking link via email after your order ships"
    ]
    predicted_answers = [
        "Returns are accepted within one month with proof of purchase",
        "Delivery usually takes about a week",
        "Shipping is free when you spend more than fifty dollars",
        "We take all major credit cards and PayPal",
        "A tracking number will be emailed to you once your package ships"
    ]
    results = evaluate_dataset(questions, expected_answers, predicted_answers)
    return results