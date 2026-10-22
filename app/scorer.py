from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the model once when the app starts
model = SentenceTransformer('all-MiniLM-L6-v2')

def score_with_local_model(expected: str, predicted: str) -> dict:
    """
    Compare two sentences and return a similarity score.
    Uses BERT embeddings and cosine similarity.
    """
    # Convert sentences to vectors
    embeddings = model.encode([expected, predicted])
    
    # Calculate how similar the two vectors are
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Convert to score out of 10
    score = round(float(similarity) * 10, 2)
    
    # Pass or fail threshold
    status = "passed" if score >= 7.0 else "failed"
    
    return {
        "score": score,
        "similarity": round(float(similarity), 4),
        "status": status,
        "method": "local"
    }


def evaluate_dataset(questions: list, expected_answers: list, predicted_answers: list) -> dict:
    """
    Evaluate a full dataset and return a complete report.
    """
    results = []
    total_score = 0
    passed = 0
    failed = 0

    for i, (question, expected, predicted) in enumerate(zip(questions, expected_answers, predicted_answers)):
        # Score each answer
        result = score_with_local_model(expected, predicted)
        
        results.append({
            "id": i + 1,
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "score": result["score"],
            "status": result["status"],
            "similarity": result["similarity"]
        })
        
        total_score += result["score"]
        if result["status"] == "passed":
            passed += 1
        else:
            failed += 1

    overall_score = round(total_score / len(results), 2) if results else 0

    return {
        "overall_score": overall_score,
        "total_questions": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{round((passed / len(results)) * 100, 1)}%",
        "results": results
    }