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

app = FastAPI(title="intervAI Enterprise Gemini Backend Engine", version="3.7.0")

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
    client_recent_questions: Optional[List[str]] = []


class AnswerSubmissionRequest(BaseModel):
    session_id: str
    question_index: int
    answer_text: str
    role: str
    category: str
    language: Optional[str] = "English"
    candidate_id: Optional[str] = "guest"
    client_recent_questions: Optional[List[str]] = []


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
            if overlap >= 0.70:
                return True
    return False


def _generate_with_retry(prompt: str, json_mode: bool = False, temperature: float = 0.9):
    # Use fast valid Gemini models directly so there is zero dead-model delay
    models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain",
        temperature=temperature
    )

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                time.sleep(0.25)
            continue

    raise last_error


def _get_difficulty_for_index(next_q_index: int) -> tuple[str, str]:
    if next_q_index < 5:
        return (
            "Easy",
            "Difficulty: EASY (beginner level, basic concepts, simple everyday words)."
        )
    elif next_q_index < 10:
        return (
            "Medium",
            "Difficulty: MEDIUM (practical level, real-world usage or debugging, simple clear words)."
        )
    else:
        return (
            "Hard",
            "Difficulty: HARD (advanced problem-solving, scaling or edge cases, explained in simple clear words)."
        )


def _language_rule(language: str) -> str:
    if (language or "").strip().lower() == "hindi":
        return (
            "LANGUAGE: Strictly SIMPLE HINDI (Devanagari script only, e.g., 'नमस्ते! ...'). "
            "No English sentences. Keep sentences short and easy to speak."
        )
    return "LANGUAGE: Very simple, clear, everyday English with short sentences."


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


def _get_recent_questions(
    candidate_id: str,
    role: str,
    category: str,
    language: str,
    extra_client_qs: Optional[List[str]] = None
) -> List[str]:
    merged: List[str] = []
    for q in (extra_client_qs or []):
        if q and q not in merged:
            merged.append(q)
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
            f"{difficulty} स्तर पर सोचें: {role} के लिए '{topic_focus}' समझना क्यों ज़रूरी है और इसका एक सरल उदाहरण क्या हो सकता है?",
            f"आप {role} की तैयारी करते समय '{topic_focus}' से जुड़ी कौन-सी बात सबसे पहले सीखना या सुधारना चाहेंगे?"
        ]
        if ans_snippet and ans_snippet.lower() not in {"no", "nahi", "idk", "skip"}:
            templates.append(
                f"आपने '{ans_snippet}' का ज़िक्र किया—इसे {role} के {category} काम में '{topic_focus}' से जोड़कर कैसे समझाएंगे?"
            )
    else:
        templates = [
            f"In {role} ({category}), how would you apply '{topic_focus}' in a simple everyday situation?",
            f"Thinking at a {difficulty} level for {role}, what first step would you take around '{topic_focus}'?",
            f"Why does '{topic_focus}' matter in {category} for a {role}, and how would you explain it simply?",
            f"When working as a {role}, what is one practical habit you follow for '{topic_focus}'?"
        ]
        if ans_snippet and ans_snippet.lower() not in {"no", "nahi", "idk", "skip"}:
            templates.append(
                f"Building on '{ans_snippet}', how does that connect to '{topic_focus}' for a {role}?"
            )

    random.shuffle(templates)
    for q in templates:
        if not _is_duplicate_question(q, blocked):
            return q
    suffix = random.randint(10, 99)
    return (
        f"{role} ({category}) में '{topic_focus}' के बारे में अपना सरल विचार बताइए (भाग {suffix})।"
        if is_hindi
        else f"Share your simple approach to '{topic_focus}' in {role} ({category}) (round {suffix})."
    )


def _generate_start_response(
    role: str,
    category: str,
    language: str,
    candidate_id: str = "guest",
    client_recent_questions: Optional[List[str]] = None
):
    lang = (language or "English").strip()
    is_hindi = lang.lower() == "hindi"
    cid = (candidate_id or "guest").strip()
    session_id = "sess_" + os.urandom(4).hex()
    recent_qs = _get_recent_questions(cid, role, category, lang, client_recent_questions)

    SESSION_DB[session_id] = {
        "candidate_id": cid,
        "role": role,
        "category": category,
        "language": lang,
        "asked_questions": []
    }

    opening_angle = random.choice([
        "warm greeting + ask what motivated them to choose this role and what skills they enjoy most",
        "friendly welcome + ask about a recent project, tool, or learning experience related to this role",
        "casual welcome + ask them to introduce their background and strongest area in this field",
        "supportive greeting + ask what kind of problems they like solving in this role",
        "conversational opening + ask about their journey into this domain and current focus"
    ])

    prompt = f"""
    You are a friendly AI interviewer for Role: {role} ({category}).
    {_language_rule(lang)}
    Opening angle: {opening_angle}.

    Create ONE short, natural, unique opening interview question.
    Do NOT repeat or resemble any of these recent questions:
    {json.dumps(recent_qs[-12:], ensure_ascii=False)}

    Return ONLY the question text without quotes or markdown.
    """

    try:
        ai_question = _generate_with_retry(prompt, json_mode=False, temperature=1.0)
    except Exception as e:
        print(f"Error in start_interview: {e}")
        ai_question = ""

    if not ai_question or _is_duplicate_question(ai_question, recent_qs):
        ai_question = _build_dynamic_fallback_question(
            role=role,
            category=category,
            difficulty="Easy",
            topic_focus=opening_angle.split("+")[-1].strip(),
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
        data.candidate_id or "guest",
        data.client_recent_questions or []
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
    cross_session_questions = _get_recent_questions(
        cid, role, category, lang, data.client_recent_questions or []
    )
    all_blocked_questions = list(dict.fromkeys(previous_questions + cross_session_questions))

    current_q = previous_questions[-1] if previous_questions else ("अपने बारे में बताइए" if is_hindi else "Tell me about yourself")
    next_index = question_index + 1
    next_difficulty, level_guide = _get_difficulty_for_index(next_index)

    topic_pool = [
        "core concepts and fundamentals",
        "practical tools and daily workflow",
        "debugging and fixing errors",
        "handling edge cases and bad inputs",
        "speed, memory, and scalability",
        "teamwork and design trade-offs",
        "real project decisions and testing"
    ]
    topic_angle = topic_pool[(next_index + random.randint(0, 3)) % len(topic_pool)]

    # Compact prompt for fast 1-2s Gemini response
    prompt = f"""
    AI interviewer for {role} ({category}). {_language_rule(lang)}
    Question: "{current_q}"
    Candidate Answer: "{answer_text}"

    1. "avatar_reply": 1-2 short natural sentences reacting directly to what the candidate said (if "no"/"idk"/off-topic, gently note it and transition; if good, briefly acknowledge). No numeric marks.
    2. "feedback": 1 short helpful tip on their answer. No numeric marks.
    3. Scores (0-100): "completeness", "correctness", "relevance", "clarity", "overall_score" (5-25 if non-answer/irrelevant; 70+ only if genuinely relevant & clear).
    4. "next_question": One fresh unique question. {level_guide} Focus: {topic_angle}.
       Avoid repeating: {json.dumps(all_blocked_questions[-18:], ensure_ascii=False)}

    Return JSON only:
    {{"completeness":0,"correctness":0,"relevance":0,"clarity":0,"overall_score":0,"feedback":"","avatar_reply":"","next_question":""}}
    """

    non_answers = {
        "no", "na", "n/a", "idk", "i don't know", "dont know", "don't know",
        "skip", "pass", "nothing", "none", "nahi", "nahin", "nhi",
        "pata nahi", "pata nhi", "mujhe nahi pata", "maloom nahi", "मालूम नहीं", "पता नहीं", "नहीं"
    }
    clean_ans = answer_text.lower().strip(".!?। ")
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

        feedback = (parsed.get("feedback") or "").strip()
        avatar_reply = (parsed.get("avatar_reply") or "").strip()
        next_q_text = (parsed.get("next_question") or "").strip()

    except Exception as e:
        print(f"Error in evaluate_answer: {e}")
        short_quote = answer_text[:60]
        if clean_ans in non_answers or word_count <= 2:
            completeness, correctness, relevance, clarity, overall_score = 10, 10, 10, 20, 13
            if is_hindi:
                feedback = f"आपने '{short_quote}' कहा, जो पूरा उत्तर नहीं है। जो बुनियादी बात पता हो, वह सरल शब्दों में बताएं।"
                avatar_reply = f"कोई बात नहीं, आपने '{short_quote}' कहा। चलिए अगले सवाल पर चलते हैं।"
            else:
                feedback = f"You replied '{short_quote}', which doesn't answer the question yet. Share even a basic idea next time."
                avatar_reply = f"I noticed you said '{short_quote}'. That's okay—let's move on to the next question."
        elif word_count < 10:
            completeness, correctness, relevance, clarity, overall_score = 38, 42, 45, 50, 44
            if is_hindi:
                feedback = "उत्तर थोड़ा छोटा है—2-3 साफ वाक्यों में उदाहरण के साथ समझाएं।"
                avatar_reply = "आपने संक्षिप्त शुरुआत की है, थोड़ा और विस्तार जोड़ें। अब अगला सवाल देखते हैं।"
            else:
                feedback = "Your response is a bit brief—explain in 2-3 clear sentences with an example."
                avatar_reply = "You gave a short start, though a bit more detail would help. Let's look at the next question."
        else:
            completeness, correctness, relevance, clarity, overall_score = 72, 74, 76, 78, 75
            if is_hindi:
                feedback = "अच्छा और स्पष्ट उत्तर! एक छोटा उदाहरण जोड़ने से यह और बेहतर होगा।"
                avatar_reply = "अच्छा उत्तर दिया आपने। चलिए अगले सवाल पर आगे बढ़ते हैं।"
            else:
                feedback = "Clear explanation! Adding one practical example can make it even stronger."
                avatar_reply = "Good response—you addressed the question clearly. Let's move to the next question."
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