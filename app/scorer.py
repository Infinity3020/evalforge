from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

def score_with_local_model(expected: str, predicted: str) -> dict:
    embeddings = model.encode([expected, predicted])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    score = round(float(similarity) * 10, 2)
    return {
        "score": score,
        "similarity": round(float(similarity), 4),
        "status": "passed" if score >= 7.0 else "failed",
        "method": "local"
    }

def score_with_openai(expected: str, predicted: str, api_key: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=api_key)
    prompt = f'Score how well the AI answer matches the expected answer, 0 to 10.\n\nExpected: {expected}\nAI Answer: {predicted}\n\nRespond only with JSON like: {{"score": 8.5, "reason": "..."}}'
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    result = json.loads(response.choices[0].message.content)
    score = round(float(result["score"]), 2)
    return {
        "score": score,
        "similarity": round(score / 10, 4),
        "status": "passed" if score >= 7.0 else "failed",
        "method": "openai",
        "reason": result.get("reason", "")
    }

def score_with_claude(expected: str, predicted: str, api_key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f'Score how well the AI answer matches the expected answer, 0 to 10.\n\nExpected: {expected}\nAI Answer: {predicted}\n\nRespond only with JSON like: {{"score": 8.5, "reason": "..."}}'
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    result = json.loads(message.content[0].text)
    score = round(float(result["score"]), 2)
    return {
        "score": score,
        "similarity": round(score / 10, 4),
        "status": "passed" if score >= 7.0 else "failed",
        "method": "claude",
        "reason": result.get("reason", "")
    }

def score_with_ensemble(expected: str, predicted: str, methods: list, api_key: str = "") -> dict:
    scores = []
    methods_used = []

    if "local" in methods:
        result = score_with_local_model(expected, predicted)
        scores.append(result["score"])
        methods_used.append("local")

    if "openai" in methods and api_key:
        result = score_with_openai(expected, predicted, api_key)
        scores.append(result["score"])
        methods_used.append("openai")

    if "claude" in methods and api_key:
        result = score_with_claude(expected, predicted, api_key)
        scores.append(result["score"])
        methods_used.append("claude")

    if not scores:
        result = score_with_local_model(expected, predicted)
        scores.append(result["score"])
        methods_used.append("local")

    final_score = round(sum(scores) / len(scores), 2)
    return {
        "score": final_score,
        "similarity": round(final_score / 10, 4),
        "status": "passed" if final_score >= 7.0 else "failed",
        "method": f"ensemble({'+'.join(methods_used)})",
        "individual_scores": dict(zip(methods_used, scores))
    }

def evaluate_dataset(questions: list, expected_answers: list, predicted_answers: list, method: str = "local", api_key: str = "") -> dict:
    results = []
    total_score = 0
    passed = 0
    failed = 0

    for i, (question, expected, predicted) in enumerate(zip(questions, expected_answers, predicted_answers)):
        if "+" in method:
            methods_list = method.split("+")
            result = score_with_ensemble(expected, predicted, methods_list, api_key)
        elif method == "openai":
            result = score_with_openai(expected, predicted, api_key)
        elif method == "claude":
            result = score_with_claude(expected, predicted, api_key)
        else:
            result = score_with_local_model(expected, predicted)

        results.append({
            "id": i + 1,
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "score": result["score"],
            "status": result["status"],
            "similarity": result["similarity"],
            "method": result["method"]
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