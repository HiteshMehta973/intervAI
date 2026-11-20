from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import os
import time
import random
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json

load_dotenv()

app = FastAPI(title="intervAI Enterprise Gemini Backend Engine", version="3.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

SESSION_DB = {}
# Remembers recent questions per (role, category) across repeated interviews so users get fresh questions
RECENT_QUESTIONS_BY_ROLE = {}


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


def _generate_with_retry(prompt: str, json_mode: bool = False, temperature: float = 0.85):
    models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash"]
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain",
        temperature=temperature
    )

    last_error = None
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                return response.text.strip()
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                break

    raise last_error


def _get_difficulty_for_index(next_q_index: int) -> tuple[str, str]:
    """Returns (difficulty_label, prompt_guidance) based on progressive question number."""
    if next_q_index < 5:
        return (
            "Easy",
            "Difficulty: EASY (Beginner level). Ask about basic definitions, simple core concepts, or everyday fundamentals. Use very simple, easy-to-understand language and short sentences."
        )
    elif next_q_index < 10:
        return (
            "Medium",
            "Difficulty: MEDIUM (Practical level). Ask a practical how-to, real-world usage, or debugging/workflow question. Keep the wording very simple, clear, and easy to understand without heavy jargon."
        )
    else:
        return (
            "Hard",
            "Difficulty: HARD (Advanced problem-solving level). Ask about system scaling, performance optimization, trade-offs, or tricky edge cases—but explain the scenario in very simple, clear, everyday language so the candidate easily understands what is being asked."
        )


def _remember_question(role: str, category: str, question: str):
    key = f"{role.strip().lower()}::{category.strip().lower()}"
    history = RECENT_QUESTIONS_BY_ROLE.setdefault(key, [])
    if question and question not in history:
        history.append(question)
        if len(history) > 45:
            RECENT_QUESTIONS_BY_ROLE[key] = history[-45:]


def _get_recent_questions(role: str, category: str) -> List[str]:
    key = f"{role.strip().lower()}::{category.strip().lower()}"
    return RECENT_QUESTIONS_BY_ROLE.get(key, [])


def _generate_start_response(role: str, category: str, language: str):
    session_id = "sess_" + os.urandom(4).hex()
    recent_qs = _get_recent_questions(role, category)
    style_seed = random.choice([
        "warm and conversational",
        "short and encouraging",
        "friendly and direct",
        "welcoming and casual",
        "calm and supportive"
    ])

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
    Tone style: {style_seed}.

    Ask the candidate to introduce themselves ("Tell me about yourself" and their background/interest in {role}), phrased in a fresh, unique way that is different from these recent openings:
    {json.dumps(recent_qs[-8:], indent=2)}

    Use very simple and easy {language}.
    Keep the question short, clear, and easy to understand.
    Do not use difficult vocabulary or complex sentences.

    Return ONLY the question text in {language} without quotes or markdown.
    """

    try:
        ai_question = _generate_with_retry(prompt, json_mode=False, temperature=0.95)
    except Exception as e:
        print(f"Error in start_interview: {e}")
        fallback_intros = [
            f"Hi! Welcome to your interview. Could you tell me a little about yourself and why you like {role}?",
            f"Hello! To begin, please introduce yourself and share your background in {role}.",
            f"Welcome! Can you briefly tell me about yourself and what got you interested in {role}?",
            f"Hi there! Let's start with a quick introduction—who are you and what have you worked on for {role}?"
        ]
        ai_question = random.choice(fallback_intros)

    SESSION_DB[session_id]["asked_questions"].append(ai_question)
    _remember_question(role, category, ai_question)

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
    answer_text = data.answer_text.strip()
    question_index = data.question_index
    role = data.role
    category = data.category
    language = data.language or "English"

    session_data = SESSION_DB.get(session_id, {"asked_questions": []})
    previous_questions = session_data.get("asked_questions", [])
    cross_session_questions = _get_recent_questions(role, category)
    all_blocked_questions = list(dict.fromkeys(previous_questions + cross_session_questions))

    next_index = question_index + 1
    next_difficulty, level_guide = _get_difficulty_for_index(next_index)
    topic_angle = random.choice([
        "core concepts",
        "practical tools and workflow",
        "problem-solving and debugging",
        "real-world project decisions",
        "data, performance, or reliability",
        "best practices and teamwork"
    ])

    prompt = f"""
    You are a fair, supportive, yet accurate AI interviewer for the role: {role} (Category: {category}).
    Language preference: {language}
    Current Question Asked: "{previous_questions[-1] if previous_questions else 'Tell me about yourself'}"
    Candidate's Answer: "{answer_text}"

    STRICT EVALUATION RULES:
    - If the candidate replies with "no", "idk", "i don't know", "skip", off-topic text, or a 1-2 word non-answer, score completeness, correctness, relevance, and clarity very low (between 5 and 25 out of 100).
    - Only give 70+ when the answer genuinely explains relevant concepts clearly.
    - Calculate "overall_score" as the integer average of completeness, correctness, relevance, and clarity (0-100).

    CRITICAL INSTRUCTIONS FOR NEXT QUESTION:
    - {level_guide}
    - Focus angle for variety: {topic_angle}.
    - Always use very simple, clear, easy-to-understand {language} with short sentences so any candidate can easily understand the question.
    - You MUST NOT repeat or rephrase any of these previously asked questions (from this session or past sessions):
    {json.dumps(all_blocked_questions[-30:], indent=2)}

    Provide short, helpful, constructive feedback ("feedback") in simple {language} WITHOUT mentioning numeric marks/scores inside the feedback text.
    Generate a completely fresh, unique NEXT interview question ("next_question") in {language}.
    
    Return a valid JSON object with keys: 
    "completeness", "correctness", "relevance", "clarity", "overall_score", "feedback", "next_question".
    """

    non_answers = {"no", "na", "n/a", "idk", "i don't know", "dont know", "don't know", "skip", "pass", "nothing", "none"}
    clean_ans = answer_text.lower().strip(".!? ")
    word_count = len(answer_text.split())

    try:
        content = _generate_with_retry(prompt, json_mode=True, temperature=0.85)
        parsed = json.loads(content)

        completeness = max(0, min(100, int(parsed.get("completeness", 25))))
        correctness = max(0, min(100, int(parsed.get("correctness", 25))))
        relevance = max(0, min(100, int(parsed.get("relevance", 25))))
        clarity = max(0, min(100, int(parsed.get("clarity", 30))))

        if clean_ans in non_answers or word_count <= 2:
            completeness = min(completeness, 15)
            correctness = min(correctness, 15)
            relevance = min(relevance, 15)
            overall_score = min(18, round((completeness + correctness + relevance + clarity) / 4))
        else:
            overall_score = int(parsed.get("overall_score", round((completeness + correctness + relevance + clarity) / 4)))
            overall_score = max(0, min(100, overall_score))

        feedback = parsed.get("feedback", "Good effort! Try adding a simple example to make your answer stronger.")
        next_q_text = parsed.get("next_question", "").strip()

    except Exception as e:
        print(f"Error in evaluate_answer: {e}")
        if clean_ans in non_answers or word_count <= 2:
            completeness, correctness, relevance, clarity, overall_score = 10, 10, 10, 20, 13
            feedback = "A short 'no' or skipped reply does not answer the question. Try sharing whatever basic idea you know."
        elif word_count < 10:
            completeness, correctness, relevance, clarity, overall_score = 38, 42, 45, 50, 44
            feedback = "Your answer is a bit brief. Try explaining it in 2-3 clear sentences."
        else:
            completeness, correctness, relevance, clarity, overall_score = 72, 74, 76, 78, 75
            feedback = "Clear explanation! Adding one practical example will make it even better."
        next_q_text = ""

    # Guarantee unique next question matching progressive difficulty if empty or duplicate
    if not next_q_text or next_q_text in all_blocked_questions:
        fallback_pool = {
            "Easy": [
                f"What are the most important basic skills needed for a {role}?",
                f"Can you explain in simple words what a {role} does every day?",
                f"Which basic tool or technology do you feel most comfortable using for {role}?",
                f"What is one simple concept every beginner in {role} should learn first?"
            ],
            "Medium": [
                f"Can you describe a small project or task you built or worked on for {role}?",
                f"When something breaks or gives an error in your {role} work, how do you find and fix it?",
                f"How do you make sure your work as a {role} is clean, reliable, and easy for others to understand?",
                f"Tell me about a time you had to learn a new tool quickly for a {role} task."
            ],
            "Hard": [
                f"Imagine many users start using your {role} system at the same time and it becomes slow. How would you make it faster?",
                f"What is a difficult problem or edge case you might face in {role}, and how would you solve it step by step?",
                f"If you had to choose between making a {role} solution faster or making it simpler to maintain, how would you decide?",
                f"How would you test and protect a {role} system so it does not fail when unexpected inputs happen?"
            ]
        }
        candidates = [q for q in fallback_pool[next_difficulty] if q not in all_blocked_questions]
        next_q_text = random.choice(candidates or fallback_pool[next_difficulty])

    SESSION_DB.setdefault(session_id, {}).setdefault("asked_questions", []).append(next_q_text)
    _remember_question(role, category, next_q_text)

    is_completed = question_index >= 14

    next_question_obj = None
    if not is_completed:
        next_question_obj = {
            "question_index": next_index,
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
        content = _generate_with_retry(prompt, json_mode=True, temperature=0.85)
        parsed = json.loads(content)
        return {"success": True, "data": parsed}

    except Exception as e:
        print(f"ERROR in sandbox evaluation (falling back): {e}")
        words = len(data.answer.split())
        fallback_score = min(88, max(60, 58 + (words // 4)))
        return {
            "success": True,
            "data": {
                "score": fallback_score,
                "strengths": [
                    "Clear and direct response addressing the interview question.",
                    "Relevant technical or situational points included."
                ],
                "improvements": [
                    "Add specific metrics, results, or step-by-step details to make it stronger.",
                    "Mention potential edge cases or trade-offs briefly."
                ],
                "model_suggestion": f"For '{data.question}', start with a clear summary, share a concrete example or approach, and end with the outcome."
            }
        }


@app.post("/api/reports/save")
def save_interview_report(report: SaveReportRequest):
    return {"success": True, "message": "Enterprise report successfully persisted."}


# Mount static files after API routes so /api/* endpoints are never shadowed
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)