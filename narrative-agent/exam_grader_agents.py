import os
from pathlib import Path
import time
import json
import statistics
import pickle
import re

from dotenv import load_dotenv
import openai
import pdfplumber

# Optional imports for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Load environment variables from .env (if present)
load_dotenv()

# Configuration: read API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")


def is_mock_mode_enabled() -> bool:
    return os.getenv("MOCK_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _estimate_question_count(questions: str) -> int:
    numbered = re.findall(r"(?im)^\s*(?:q\s*\d+|question\s*\d+|\d+[\).:])", questions or "")
    if numbered:
        return min(max(len(numbered), 1), 10)
    return 3


def _mock_grade_exam(questions: str) -> dict:
    count = _estimate_question_count(questions)
    scores = []
    for i in range(1, count + 1):
        score = float(7 + (i % 2))
        scores.append(
            {
                "question_id": i,
                "score": score,
                "feedback": "Mock feedback: argument is coherent; include one stronger supporting detail."
            }
        )

    overall = round(sum(item["score"] for item in scores) / len(scores), 2) if scores else 0.0
    return {
        "scores": scores,
        "overall_score": overall,
        "general_feedback": "Mock result for deployment testing. Depth and evidence could be improved.",
        "metrics": {
            "num_questions": len(scores),
            "score_variance": statistics.variance([s["score"] for s in scores]) if len(scores) > 1 else 0.0,
            "score_mean": overall,
            "replicability_seed": 42,
            "response_time_seconds": 0.01,
            "temperature_used": 0.0,
            "historical_runs": 0,
            "MAE_vs_mean": 0.0,
            "RMSE_vs_mean": 0.0,
            "mock_mode": True
        }
    }

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0  # seconds
METRICS_HISTORY_FILE = "grading_metrics_history.pkl"

# Define the function schema for structured output
def get_functions():
    return [
        {
            "name": "generate_exam_responses",
            "description": (
                "Grade each question and provide feedback according to the rubric if available, "
                "otherwise generate your own criteria to score from 0-10 with constructive feedback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question_id": {"type": "integer"},
                                "score": {"type": "number"},
                                "feedback": {"type": "string"}
                            },
                            "required": ["question_id", "score", "feedback"]
                        }
                    },
                    "overall_score": {"type": "number"},
                    "general_feedback": {"type": "string"}
                },
                "required": ["scores", "overall_score", "general_feedback"]
            }
        }
    ]

# Extract text from PDF or TXT
def extract_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".pdf":
        texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return "\n".join(texts)
    else:
        return path.read_text(encoding="utf-8")

# Retry wrapper with exponential backoff
def call_with_backoff(**kwargs):
    from openai import RateLimitError, APIConnectionError, APITimeoutError
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return openai.chat.completions.create(**kwargs)
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))

# Grade the exam via OpenAI
def grade_exam(rubric: str, questions: str, responses: str) -> dict:
    start_time = time.time()

    if is_mock_mode_enabled():
        return _mock_grade_exam(questions)

    if rubric.strip():
        system_prompt = (
            "You are an exam grader. Use the rubric to assign each question a numeric score (0-10) and valuable concise feedback so the student can further understand their strengths and weaknesses of the material. "
            "Then compute the overall score as the average and provide general feedback. Return JSON."
        )
        user_prompt = f"Rubric:\n{rubric}\n\nQuestions:\n{questions}\n\nStudent Responses:\n{responses}"
    else:
        system_prompt = (
            "You are an exam grader. The rubric is not available. Use your own criteria to assign each question a numeric score (0-10) and constructive feedback. "
            "Then compute the overall score as the average and provide general feedback. Return JSON."
        )
        user_prompt = f"Questions:\n{questions}\n\nStudent Responses:\n{responses}"

    resp = call_with_backoff(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        tools=[
            {"type": "function", "function": f} for f in get_functions()
        ],
        tool_choice={"type": "function", "function": {"name": "generate_exam_responses"}},
        temperature=0.3,
        seed=42
    )

    output = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    scores = [q["score"] for q in output.get("scores", []) if isinstance(q.get("score"), (int, float))]
    elapsed = time.time() - start_time

    # Load past scores
    if os.path.exists(METRICS_HISTORY_FILE):
        with open(METRICS_HISTORY_FILE, "rb") as f:
            history = pickle.load(f)
    else:
        history = []

    history.append(scores)

    # Save updated history
    with open(METRICS_HISTORY_FILE, "wb") as f:
        pickle.dump(history, f)

    # Flatten all scores for MAE and RMSE
    all_scores = [score for run in history for score in run]
    average_score = statistics.mean(all_scores) if all_scores else 0.0
    mae = statistics.mean(abs(s - average_score) for s in all_scores) if all_scores else 0.0
    rmse = (statistics.mean((s - average_score)**2 for s in all_scores)**0.5) if all_scores else 0.0

    # Additional metrics
    output["metrics"] = {
        "num_questions": len(scores),
        "score_variance": statistics.variance(scores) if len(scores) > 1 else 0.0,
        "score_mean": statistics.mean(scores) if scores else 0.0,
        "replicability_seed": 42,
        "response_time_seconds": round(elapsed, 2),
        "temperature_used": 0.3,
        "historical_runs": len(history),
        "MAE_vs_mean": round(mae, 2),
        "RMSE_vs_mean": round(rmse, 2)
    }

    return output

# Create PDF report
def create_pdf_report(results: dict, output_path: Path):
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required for PDF generation. Install it with: pip install reportlab")
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    margin = inch
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Exam Grading Report")
    y -= 0.5 * inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Q#")
    c.drawString(margin + 1.0*inch, y, "Score")
    c.drawString(margin + 2.0*inch, y, "Feedback")
    y -= 0.3 * inch

    c.setFont("Helvetica", 11)
    for item in results.get("scores", []):
        qid = item["question_id"]
        score = item["score"]
        fb = item["feedback"]
        c.drawString(margin, y, str(qid))
        c.drawString(margin + 1.0*inch, y, f"{score:.1f}")
        text = fb.replace("\n", " ")
        c.drawString(margin + 2.0*inch, y, text[:80])
        y -= 0.4 * inch
        if y < margin:
            c.showPage()
            y = height - margin

    y -= 0.2 * inch
    c.setFont("Helvetica-Bold", 12)
    overall = results.get("overall_score", 0)
    c.drawString(margin, y, f"Overall Score: {overall:.2f}")
    y -= 0.3 * inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "General Feedback:")
    y -= 0.2 * inch
    c.setFont("Helvetica", 11)
    for line in results.get("general_feedback", "").split("\n"):
        c.drawString(margin, y, line)
        y -= 0.3 * inch
        if y < margin:
            c.showPage()
            y = height - margin

    # Add Metrics Section
    y -= 0.5 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Evaluation Metrics:")
    y -= 0.3 * inch
    c.setFont("Helvetica", 11)
    metrics = results.get("metrics", {})
    for key, value in metrics.items():
        c.drawString(margin, y, f"{key.replace('_', ' ').title()}: {value}")
        y -= 0.3 * inch
        if y < margin:
            c.showPage()
            y = height - margin

    c.save()