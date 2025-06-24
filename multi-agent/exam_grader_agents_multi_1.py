from typing import List, Dict, Any
import os
import time
import json
import pdfplumber
import librosa
import pandas as pd
from pathlib import Path
from openai import OpenAIError, RateLimitError, APIConnectionError, Timeout
from dotenv import load_dotenv
import openai

# Load environment variables from .env (if present)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0
RUBRIC_CACHE_DIR = os.path.join(os.getcwd(), "cache")
os.makedirs(RUBRIC_CACHE_DIR, exist_ok=True)

# Grading rubric for VC pitch
RUBRIC_VC = """
You are a seasoned VC pitch grader. For a {{duration}}-minute audio pitch, give each dimension a score from 1 (poor) to 10 (excellent), using the following anchors:

1. Problem Clarity  
   • 1–3: No clear problem stated, listener confused  
   • 4–6: Problem mentioned but lacks context or urgency  
   • 7–8: Problem clearly described with context  
   • 9–10: Problem statement is crisp, impactful, and immediately compelling

2. Market Evidence  
   • 1–3: No market data or vague claims  
   • 4–6: Qualitative market description, no numbers  
   • 7–8: One clear quantitative metric (TAM, growth rate)  
   • 9–10: Multiple strong data points (TAM, traction, growth) cited

3. Solution Differentiation  
   • 1–3: Solution not differentiated, generic  
   • 4–6: Mentions a unique feature but no defense  
   • 7–8: Clearly highlights one defensible advantage  
   • 9–10: Demonstrates multiple, well-justified differentiators or proprietary edge

4. Delivery & Pacing  
   • 1–3: Monotone or too fast/slow (outside 80–200 WPM), frequent long pauses (>30 %)  
   • 4–6: Understandable but some pacing issues (WPM 90–210, pauses 20–30 %)  
   • 7–8: Good pace (110–160 WPM), pauses <20 %  
   • 9–10: Engaging tone, ideal pacing (120–150 WPM), minimal pauses (<10 %)

Return valid JSON EXACTLY in this format (no extra keys):
{
  "Problem": <1–10>,
  "Market": <1–10>,
  "Solution": <1–10>,
  "Delivery": <1–10>,
  "Feedback": "<one sentence actionable feedback for each anchor>"
}
"""

# Technical agent prompt
TECHNICAL_PROMPT_TEMPLATE = """
You are a grading assistant for technical exams. Your role is to evaluate student responses based on the provided questions and, if available, the rubric.

### Grading Instructions:
- For each question:
  - Read the question and the student's answer.
  - If a rubric is provided for that question, follow it carefully to assign points based on the expected criteria.
  - If no rubric is provided, use your expert-level knowledge of the subject to assess:
    - Factual accuracy.
    - Completeness.
    - Clarity.
- Assign a score for each answer:
  - Use the point scale from the rubric, or if none is provided, use a default scale of 0-10 points.
- Provide feedback for each answer:
  - Explain why the student received the score.
  - Offer suggestions for improvement if the answer is incomplete or incorrect.

### Output Format:
Respond ONLY with raw JSON (no markdown):
{
  "question_1": {
    "score": X,
    "feedback": "..."
  },
  "question_2": {
    "score": Y,
    "feedback": "..."
  },
  ...
  "total_score": Z
}

### Rubric (if available):
{rubric_markdown}
"""

# Narrative agent prompt
NARRATIVE_PROMPT_WITH_RUBRIC = """
You are an exam grader. Use the rubric to assign each question a numeric score (0-10) and valuable concise feedback so the student can further understand their strengths and weaknesses of the material. Then compute the overall score as the average and provide general feedback. Return JSON.

### Output Format:
Respond ONLY with raw JSON (no markdown):
{
  "question_1": {
    "score": X,
    "feedback": "..."
  },
  "question_2": {
    "score": Y,
    "feedback": "..."
  },
  ...
  "total_score": Z
}
"""

NARRATIVE_PROMPT_NO_RUBRIC = """
You are an exam grader. The rubric is not available. Use your own criteria to assign each question a numeric score (0-10) and constructive feedback. Then compute the overall score as the average and provide general feedback. Return JSON.

### Output Format:
Respond ONLY with raw JSON (no markdown):
{
  "question_1": {
    "score": X,
    "feedback": "..."
  },
  "question_2": {
    "score": Y,
    "feedback": "..."
  },
  ...
  "total_score": Z
}
"""

# ========== UTILITIES ==========
def extract_pdf_to_markdown(pdf_path: str) -> str:
    def clean_text_formatting(text: str) -> str:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            s = line.strip()
            if not s:
                cleaned.append("")
            elif s[0] in ("*", "•", "·", "-"):
                cleaned.append("- " + s.lstrip("*•·-").strip())
            else:
                cleaned.append(s)
        return "\n".join(cleaned) + "\n"

    def convert_table_to_markdown(table: List[List[str]]) -> str:
        header, *rows = table
        md = "| " + " | ".join(header) + " |\n"
        md += "| " + " | ".join("--" for _ in header) + " |\n"
        for r in rows:
            md += "| " + " | ".join(cell or "" for cell in r) + " |\n"
        return md

    out = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            out += f"\n\n## Page {i}\n"
            text = page.extract_text() or ""
            out += clean_text_formatting(text)
            for tbl in page.extract_tables() or []:
                out += "\n" + convert_table_to_markdown(tbl)
    return out


def transcribe(mp3_path: str) -> str:
    """Return transcript from Whisper with simple caching logic."""
    cache_file = os.path.join(RUBRIC_CACHE_DIR, os.path.basename(mp3_path) + ".txt")
    if os.path.exists(cache_file):
        return open(cache_file, "r").read()

    with open(mp3_path, "rb") as f:
        resp = openai.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="text"
        )
    transcript = resp if isinstance(resp, str) else resp.get("text", "")
    with open(cache_file, "w") as f:
        f.write(transcript)
    return transcript


def analyze_audio(mp3_path: str) -> Dict[str, Any]:
    y, sr = librosa.load(mp3_path, sr=16000, mono=True)
    duration = len(y) / sr
    transcript = transcribe(mp3_path)
    word_count = len(transcript.split())
    wpm = word_count / (duration / 60) if duration else 0
    intervals = librosa.effects.split(y, top_db=30)
    voiced = sum((e - s) for s, e in intervals) / sr
    silence_ratio = (duration - voiced) / duration if duration else 0
    return {
        "duration": duration,
        "wpm": wpm,
        "silence_ratio": silence_ratio,
        "transcript": transcript
    }

# ========== GRADERS ==========
def call_with_backoff(**kwargs):
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return openai.chat.completions.create(**kwargs)
        except (RateLimitError, OpenAIError, APIConnectionError, Timeout):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))

def grade_exam(rubric: str, questions: str, responses: str, exam_type: str = "narrative") -> dict:
    rubric_markdown = rubric.strip() or "No rubric provided."

    if exam_type == "technical":
        user_prompt = TECHNICAL_PROMPT_TEMPLATE.format(rubric_markdown=rubric_markdown) + f"\n\nQuestions:\n{questions}\n\nStudent Responses:\n{responses}"
        system_prompt = "You are a helpful technical exam grader."

    else:  # narrative default
        if rubric.strip():
            system_prompt = NARRATIVE_PROMPT_WITH_RUBRIC
            user_prompt = f"Rubric:\n{rubric}\n\nQuestions:\n{questions}\n\nStudent Responses:\n{responses}"
        else:
            system_prompt = NARRATIVE_PROMPT_NO_RUBRIC
            user_prompt = f"Questions:\n{questions}\n\nStudent Responses:\n{responses}"

    resp = call_with_backoff(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        seed=42
    )

    raw_response = resp.choices[0].message.content
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw": raw_response}
