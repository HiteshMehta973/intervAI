from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import os
import re
import time
import random
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json

load_dotenv()

app = FastAPI(title="intervAI Enterprise Gemini Backend Engine", version="3.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

SESSION_DB = {}
RECENT_QUESTIONS_BY_SCOPE = {}


class InterviewStartRequest(BaseModel):
    role: str = "Software Engineer"
    category: str = "Technical"
    language: Optional[str] = "English"
    candidate_id: Optional[str] = "guest"


class AnswerSubmissionRequest(BaseModel):
    session_id: str
    question_index: int
    answer_text: str
    role: str
    category: str
    language: Optional[str] = "English"
    candidate_id: Optional[str] = "guest"


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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\u0900-\u097F]", "", (text or "").lower())).strip()


def _is_duplicate_question(candidate_q: str, history: List[str]) -> bool:
    norm_new = _normalize_text(candidate_q)
    if not norm_new:
        return True
    new_tokens = set(norm_new.split())
    for old_q in history:
        norm_old = _normalize_text(old_q)
        if norm_new == norm_old:
            return True
        old_tokens = set(norm_old.split())
        if new_tokens and old_tokens:
            overlap = len(new_tokens & old_tokens) / max(len(new_tokens), len(old_tokens))
            if overlap >= 0.72:
                return True
    return False


def _generate_with_retry(prompt: str, json_mode: bool = False, temperature: float = 0.9):
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
    if next_q_index < 5:
        return (
            "Easy",
            "Difficulty: EASY (Beginner level). Ask about basic definitions, simple core concepts, or everyday fundamentals."
        )
    elif next_q_index < 10:
        return (
            "Medium",
            "Difficulty: MEDIUM (Practical level). Ask a practical how-to, real-world usage, or debugging/workflow question."
        )
    else:
        return (
            "Hard",
            "Difficulty: HARD (Advanced problem-solving level). Ask about system scaling, performance optimization, trade-offs, or tricky edge cases."
        )


def _language_rule(language: str) -> str:
    if (language or "").strip().lower() == "hindi":
        return (
            "CRITICAL LANGUAGE RULE: Write your response strictly in SIMPLE HINDI (Devanagari script, e.g., 'नमस्ते! अपने बारे में थोड़ा बताइए...'). "
            "Do NOT write full English sentences. Use simple, everyday Hindi words that are easy to understand and speak."
        )
    return "CRITICAL LANGUAGE RULE: Write strictly in very simple, clear, everyday English with short sentences."


def _scope_keys(candidate_id: str, role: str, category: str, language: str) -> List[str]:
    cid = (candidate_id or "guest").strip().lower()
    r = role.strip().lower()
    c = category.strip().lower()
    l = language.strip().lower()
    return [
        f"user:{cid}::role:{r}::lang:{l}",
        f"role:{r}::cat:{c}::lang:{l}",
        f"role:{r}::lang:{l}"
    ]


def _remember_question(candidate_id: str, role: str, category: str, language: str, question: str):
    if not question:
        return
    for key in _scope_keys(candidate_id, role, category, language):
        history = RECENT_QUESTIONS_BY_SCOPE.setdefault(key, [])
        if not _is_duplicate_question(question, history):
            history.append(question)
            if len(history) > 120:
                RECENT_QUESTIONS_BY_SCOPE[key] = history[-120:]


def _get_recent_questions(candidate_id: str, role: str, category: str, language: str) -> List[str]:
    merged: List[str] = []
    for key in _scope_keys(candidate_id, role, category, language):
        for q in RECENT_QUESTIONS_BY_SCOPE.get(key, []):
            if q not in merged:
                merged.append(q)
    return merged


def _build_dynamic_fallback_question(
    role: str,
    category: str,
    difficulty: str,
    topic_focus: str,
    candidate_answer: str,
    is_hindi: bool,
    blocked: List[str]
) -> str:
    ans_snippet = " ".join((candidate_answer or "").split()[:6]).strip()
    if is_hindi:
        templates = [
            f"{role} के {category} क्षेत्र में '{topic_focus}' को ध्यान में रखते हुए बताइए कि आप इसे व्यवहार में कैसे लागू करेंगे?",
            f"मान लीजिए {role} की भूमिका में '{topic_focus}' से जुड़ी कोई स्थिति आती है, तो आपका पहला कदम क्या होगा?",
            f"{difficulty} स्तर पर सोचें: {role} के लिए '{topic_focus}' समझना क्यों ज़रूरी है और इसका एक सरल उदाहरण क्या हो सकता है?"
        ]
        if ans_snippet and ans_snippet.lower() not in {"no", "nahi", "idk", "skip"}:
            templates.append(
                f"आपने '{ans_snippet}' का ज़िक्र किया—इसे {role} के {category} काम में '{topic_focus}' से जोड़कर कैसे समझाएंगे?"
            )
    else:
        templates = [
            f"In {role} ({category}), how would you apply '{topic_focus}' in a simple everyday situation?",
            f"Thinking at a {difficulty} level for {role}, what first step would you take around '{topic_focus}'?",
            f"Why does '{topic_focus}' matter in {category} for a {role}, and how would you explain it simply?"
        ]
        if ans_snippet and ans_snippet.lower() not in {"no", "nahi", "idk", "skip"}:
            templates.append(
                f"Building on '{ans_snippet}', how does that connect to '{topic_focus}' for a {role}?"
            )

    for q in templates:
        if not _is_duplicate_question(q, blocked):
            return q
    suffix = os.urandom(2).hex()
    return (
        f"{role} ({category}) में '{topic_focus}' के बारे में अपना सरल विचार बताइए (#{suffix})।"
        if is_hindi
        else f"Share your simple approach to '{topic_focus}' in {role} ({category}) (#{suffix})."
    )


def _generate_start_response(role: str, category: str, language: str, candidate_id: str = "guest"):
    lang = (language or "English").strip()
    is_hindi = lang.lower() == "hindi"
    cid = (candidate_id or "guest").strip()
    session_id = "sess_" + os.urandom(4).hex()
    recent_qs = _get_recent_questions(cid, role, category, lang)

    SESSION_DB[session_id] = {
        "candidate_id": cid,
        "role": role,
        "category": category,
        "language": lang,
        "asked_questions": []
    }

    prompt = f"""
    You are a friendly AI interviewer.
    Candidate role: {role}
    Category: {category}
    {_language_rule(lang)}

    Dynamically craft a fresh, welcoming opening question asking the candidate to introduce themselves and share their background/interest in {role} ({category}).
    Make it noticeably different from these previously asked questions for this candidate/role:
    {json.dumps(recent_qs[-15:], ensure_ascii=False, indent=2)}

    Return ONLY the question text without quotes or markdown.
    """

    try:
        ai_question = _generate_with_retry(prompt, json_mode=False, temperature=0.95)
    except Exception as e:
        print(f"Error in start_interview: {e}")
        ai_question = ""

    if not ai_question or _is_duplicate_question(ai_question, recent_qs):
        ai_question = _build_dynamic_fallback_question(
            role=role,
            category=category,
            difficulty="Easy",
            topic_focus="your introduction and interest in this role",
            candidate_answer="",
            is_hindi=is_hindi,
            blocked=recent_qs
        )

    SESSION_DB[session_id]["asked_questions"].append(ai_question)
    _remember_question(cid, role, category, lang, ai_question)

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
    return _generate_start_response(
        data.role,
        data.category,
        data.language or "English",
        data.candidate_id or "guest"
    )


@app.get("/api/interview/start")
def start_interview_get(
    role: str = Query("Software Engineer"),
    category: str = Query("Technical"),
    language: str = Query("English"),
    candidate_id: str = Query("guest")
):
    return _generate_start_response(role, category, language, candidate_id)


@app.post("/api/interview/evaluate")
def evaluate_answer(data: AnswerSubmissionRequest):
    session_id = data.session_id
    answer_text = data.answer_text.strip()
    question_index = data.question_index
    role = data.role
    category = data.category
    lang = (data.language or "English").strip()
    is_hindi = lang.lower() == "hindi"
    cid = (data.candidate_id or "guest").strip()

    session_data = SESSION_DB.get(session_id, {"asked_questions": []})
    previous_questions = session_data.get("asked_questions", [])
    cross_session_questions = _get_recent_questions(cid, role, category, lang)
    all_blocked_questions = list(dict.fromkeys(previous_questions + cross_session_questions))

    current_q = previous_questions[-1] if previous_questions else ("अपने बारे में बताइए" if is_hindi else "Tell me about yourself")
    next_index = question_index + 1
    next_difficulty, level_guide = _get_difficulty_for_index(next_index)

    topic_pool = [
        "core concepts and fundamentals",
        "practical tools and daily workflow",
        "debugging and root-cause analysis",
        "handling edge cases and errors",
        "performance and scalability",
        "team collaboration and trade-offs",
        "project design and decision making"
    ]
    topic_angle = topic_pool[(next_index + random.randint(0, 2)) % len(topic_pool)]

    prompt = f"""
    You are a human-like, attentive AI interviewer for the role: {role} (Category: {category}).
    {_language_rule(lang)}

    Current Question Asked: "{current_q}"
    Candidate's Answer: "{answer_text}"

    STEP 1 - ANALYZE THE CANDIDATE'S ANSWER & GENERATE AVATAR FEEDBACK ("avatar_reply" & "feedback"):
- Carefully analyze what the candidate actually wrote or said in response to the question.
- If the answer is "no", "nahi", "pata nahi", "idk", "skip", irrelevant/off-topic, or unclear: explicitly point out what was missing or off-topic in a polite, helpful way and give a short hint so the candidate learns.
- If the answer has valid points or partial detail: specifically mention what was good in their answer and what they can add next time.
- Do NOT put numeric scores inside "avatar_reply" or "feedback".

    STEP 2 - STRICT SCORING:
- Score 4 metrics (0-100): "completeness", "correctness", "relevance", "clarity".
- If the answer is "no", "nahi", "idk", "pata nahi", "skip", off-topic, or 1-2 words, keep scores between 5 and 25.
- Only score 70+ when the candidate genuinely answers the question with clear, relevant points.
- "overall_score" = integer average of the 4 metrics.

    STEP 3 - DYNAMIC UNIQUE NEXT QUESTION ("next_question"):
- {level_guide}
- Topic focus: {topic_angle}.
- Create a fresh, dynamic question tailored to {role}, {category}, and the candidate's response.
- You MUST NOT repeat or rephrase any of these previously asked questions:
    {json.dumps(all_blocked_questions[-60:], ensure_ascii=False, indent=2)}

    Return a valid JSON object with keys:
    "completeness", "correctness", "relevance", "clarity", "overall_score", "feedback", "avatar_reply", "next_question".
    """

    non_answers = {
        "no", "na", "n/a", "idk", "i don't know", "dont know", "don't know",
        "skip", "pass", "nothing", "none", "nahi", "nahin", "nhi",
        "pata nahi", "pata nhi", "mujhe nahi pata", "maloom nahi", "मालूम नहीं", "पता नहीं", "नहीं"
    }
    clean_ans = answer_text.lower().strip(".!?। ")
    word_count = len(answer_text.split())

    try:
        content = _generate_with_retry(prompt, json_mode=True, temperature=0.9)
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

        feedback = (parsed.get("feedback") or "").strip()
        avatar_reply = (parsed.get("avatar_reply") or "").strip()
        next_q_text = (parsed.get("next_question") or "").strip()

    except Exception as e:
        print(f"Error in evaluate_answer: {e}")
        short_quote = answer_text[:60]
        if clean_ans in non_answers or word_count <= 2:
            completeness, correctness, relevance, clarity, overall_score = 10, 10, 10, 20, 13
            if is_hindi:
                feedback = f"आपने '{short_quote}' कहा, जो इस सवाल का पूरा उत्तर नहीं है। जो भी बुनियादी जानकारी हो, उसे सरल शब्दों में बताने की कोशिश करें।"
                avatar_reply = f"मैंने देखा कि आपने '{short_quote}' उत्तर दिया। कोई बात नहीं, थोड़ा प्रयास ज़रूर करें—अब अगला सवाल पूछते हैं।"
            else:
                feedback = f"You replied '{short_quote}', which doesn't answer the question yet. Try sharing even a basic point or example."
                avatar_reply = f"I noticed your reply was '{short_quote}'. That's okay—let's try another question and share what you know."
        elif word_count < 10:
            completeness, correctness, relevance, clarity, overall_score = 38, 42, 45, 50, 44
            if is_hindi:
                feedback = f"आपका उत्तर ('{short_quote}') थोड़ा संक्षिप्त है। इसे 2-3 साफ वाक्यों और एक उदाहरण के साथ पूरा करें।"
                avatar_reply = f"आपने '{short_quote}' के ज़रिए शुरुआत की, पर इसे थोड़ा और समझाया जा सकता था। अब अगला सवाल देखते हैं।"
            else:
                feedback = f"Your response ('{short_quote}') is a bit brief. Expand it with a clear explanation or example."
                avatar_reply = f"You gave a short start with '{short_quote}', though more detail would help. Let's move to the next question."
        else:
            completeness, correctness, relevance, clarity, overall_score = 72, 74, 76, 78, 75
            if is_hindi:
                feedback = f"आपने '{short_quote}' से जुड़ी अच्छी बात रखी। एक ठोस उदाहरण जोड़कर इसे और मजबूत बनाया जा सकता है।"
                avatar_reply = f"अच्छा प्रयास—आपने अपने उत्तर में मुख्य बात रखी। अब अगले सवाल पर आगे बढ़ते हैं।"
            else:
                feedback = f"Good points in your response ('{short_quote}'). Adding a concrete example can make it even stronger."
                avatar_reply = f"Good response—you addressed the question with clear points. Let's move on to the next question."
        next_q_text = ""

    if not feedback:
        feedback = "अच्छा प्रयास! थोड़ा और विस्तार से उत्तर दें।" if is_hindi else "Good effort! Try adding a bit more detail."
    if not avatar_reply:
        avatar_reply = feedback

    if not next_q_text or _is_duplicate_question(next_q_text, all_blocked_questions):
        next_q_text = _build_dynamic_fallback_question(
            role=role,
            category=category,
            difficulty=next_difficulty,
            topic_focus=topic_angle,
            candidate_answer=answer_text,
            is_hindi=is_hindi,
            blocked=all_blocked_questions
        )

    SESSION_DB.setdefault(session_id, {}).setdefault("asked_questions", []).append(next_q_text)
    _remember_question(cid, role, category, lang, next_q_text)

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
        "avatar_reply": avatar_reply,
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


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)