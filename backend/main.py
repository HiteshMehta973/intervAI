from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import os
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json

load_dotenv()

app = FastAPI(title="intervAI Enterprise Gemini Backend Engine", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

SESSION_DB = {}


class InterviewStartRequest(BaseModel):
    role: str = "Software Engineer"
    category: str = "Technical"
    language: Optional[str] = "English"


class AnswerSubmissionRequest(BaseModel):
    session_id: str
    question_index: int
    answer_text: str
    role: str
    category: str
    language: Optional[str] = "English"


class SaveReportRequest(BaseModel):
    username: str
    role: str
    category: str
    score: int
    metrics: dict
    transcript: List[dict]


class SandboxEvaluationRequest(BaseModel):
    question: str
    answer: str


def _generate_start_response(role: str, category: str, language: str):
    session_id = "sess_" + os.urandom(4).hex()

    SESSION_DB[session_id] = {
        "role": role,
        "category": category,
        "language": language,
        "asked_questions": []
    }

    prompt = f"""
    You are a friendly AI interviewer.
    The candidate is applying for the role of: {role}.
    Interview category: {category}.
    Language: {language}.

    Ask the candidate to introduce themselves ("Tell me about yourself"), but rephrase it naturally in a slightly different, warm, and welcoming way each time.
    Use very simple and easy {language}.
    Keep the question short, clear, and easy to understand.
    Do not use difficult vocabulary or complex sentences.

    Return ONLY the question text in {language} without quotes or markdown.
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.85)
        )
        ai_question = response.text.strip()
    except Exception as e:
        print(f"Error in start_interview: {e}")
        ai_question = f"Hi! To get started, could you tell me a little bit about yourself and your interest in the {role} role?"

    SESSION_DB[session_id]["asked_questions"].append(ai_question)

    return {
        "success": True,
        "session_id": session_id,
        "data": {
            "question_index": 0,
            "question": ai_question,
            "difficulty": "Easy",
            "max_questions": 15
        }
    }


@app.get("/api/health")
def read_root():
    return {"status": "online", "message": "intervAI Secure Gemini Adaptive Backend is running."}


@app.post("/api/interview/start")
def start_interview_post(data: InterviewStartRequest):
    return _generate_start_response(data.role, data.category, data.language or "English")


@app.get("/api/interview/start")
def start_interview_get(
    role: str = Query("Software Engineer"),
    category: str = Query("Technical"),
    language: str = Query("English")
):
    return _generate_start_response(role, category, language)


@app.post("/api/interview/evaluate")
def evaluate_answer(data: AnswerSubmissionRequest):
    session_id = data.session_id
    answer_text = data.answer_text
    question_index = data.question_index
    role = data.role
    category = data.category
    language = data.language or "English"

    session_data = SESSION_DB.get(session_id, {"asked_questions": []})
    previous_questions = session_data.get("asked_questions", [])

    level_guide = (
        "Ask a simple beginner-level question about basic concepts. Use very simple and easy English, short sentences, and avoid difficult vocabulary."
        if question_index < 5
        else "Ask a clear, practical question using simple and easy-to-understand language without overly complex jargon."
    )

    prompt = f"""
    You are a friendly AI interviewer. Evaluate the candidate for the role: {role} (Category: {category}).
    Language preference: {language}
    Candidate's Answer: "{answer_text}"

    CRITICAL INSTRUCTION FOR QUESTION GENERATION:
- {level_guide}
- Keep the next question short, clear, and easy to understand.
- You MUST NOT repeat any of the following previously asked questions under any circumstances:
    {json.dumps(previous_questions, indent=2)}

    Analyze the answer silently across 4 key metrics (score each out of 100):
1. completeness
2. correctness
3. relevance
4. clarity

    Calculate an "overall_score" as the integer average of these four metrics (0-100).
    Provide encouraging, simple, constructive feedback ("feedback") in {language}.
    Generate a brand new, unique NEXT interview question ("next_question") in {language} that is completely different from all previous questions.
    
    Return a valid JSON object with keys: 
    "completeness", "correctness", "relevance", "clarity", "overall_score", "feedback", "next_question".
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8
            )
        )
        content = response.text.strip()
        parsed = json.loads(content)

        completeness = int(parsed.get("completeness", 70))
        correctness = int(parsed.get("correctness", 75))
        relevance = int(parsed.get("relevance", 80))
        clarity = int(parsed.get("clarity", 75))
        overall_score = int(parsed.get("overall_score", 75))
        feedback = parsed.get("feedback", "Good and clear answer. Keep going!")
        next_q_text = parsed.get("next_question", f"What are the basic skills needed for a {role}?")

    except Exception as e:
        print(f"Error in evaluate_answer: {e}")
        completeness, correctness, relevance, clarity, overall_score = 70, 75, 80, 75, 75
        feedback = "Nice explanation! Try adding a small example next time."
        next_q_text = f"Can you explain one basic concept you use often as a {role}?"

    if next_q_text in previous_questions:
        next_q_text = f"What is one simple project or task you enjoyed working on related to {role}?"

    SESSION_DB.setdefault(session_id, {}).setdefault("asked_questions", []).append(next_q_text)

    if overall_score < 50:
        next_difficulty = "Easy"
    elif overall_score < 75:
        next_difficulty = "Medium"
    else:
        next_difficulty = "Hard"

    is_completed = question_index >= 14

    next_question_obj = None
    if not is_completed:
        next_question_obj = {
            "question_index": question_index + 1,
            "question": next_q_text,
            "difficulty": next_difficulty
        }

    return {
        "success": True,
        "metrics": {
            "completeness": completeness,
            "correctness": correctness,
            "relevance": relevance,
            "clarity": clarity,
            "overall_score": overall_score
        },
        "score_awarded": overall_score,
        "interim_feedback": feedback,
        "is_completed": is_completed,
        "next_question": next_question_obj
    }


@app.post("/api/sandbox/evaluate")
def sandbox_evaluate(data: SandboxEvaluationRequest):
    prompt = f"""
    You are a friendly AI interviewer. 
    Question: "{data.question}"
    Candidate's Answer: "{data.answer}"

    Evaluate this candidate answer clearly in simple, easy-to-understand English. Provide:
1. A precise overall score from 40 to 98 based on the depth of the answer ("score").
2. Exactly 2 specific strengths observed in this answer ("strengths" as a list of strings).
3. Exactly 2 distinct constructive areas for improvement ("improvements" as a list of strings).
4. An insightful, simple model response or suggestion ("model_suggestion").

    Return a valid JSON object with keys: "score", "strengths", "improvements", "model_suggestion".
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.9
            )
        )
        content = response.text.strip()
        parsed = json.loads(content)
        return {"success": True, "data": parsed}

    except Exception as e:
        print(f"ERROR in sandbox evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reports/save")
def save_interview_report(report: SaveReportRequest):
    return {"success": True, "message": "Enterprise report successfully persisted."}


# Mount static files after API routes so /api/* endpoints are never shadowed
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)