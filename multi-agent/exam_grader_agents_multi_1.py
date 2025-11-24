from typing import List, Dict, Any
import os
import time
import json
import pdfplumber
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
