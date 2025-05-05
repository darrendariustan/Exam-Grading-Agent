import os
from pathlib import Path
from dotenv import load_dotenv

from exam_grader_agents import extract_text, grade_exam, create_pdf_report

# 1) Load environment variables from .env
load_dotenv()
# Ensure the OPENAI_API_KEY environment variable is set
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in environment"

# 2) Define file paths relative to this script
BASE_DIR = Path(__file__).parent
EXAM_OUTPUTS = BASE_DIR / "exam_outputs"
STUDENT_OUTPUTS = BASE_DIR / "student_outputs"

RUBRIC_FILE    = EXAM_OUTPUTS / "rubric.txt"
QUESTIONS_FILE = EXAM_OUTPUTS / "exam1.txt"
# Adjust to the actual student PDF filename
RESPONSE_PDF   = STUDENT_OUTPUTS / "exam1_student2.pdf"
OUTPUT_PDF     = BASE_DIR / "graded_report_4.pdf"

# 3) Extract text from inputs
rubric_text    = extract_text(RUBRIC_FILE)
questions_text = extract_text(QUESTIONS_FILE)
response_text  = extract_text(RESPONSE_PDF)

# 4) Grade the exam
results = grade_exam(rubric_text, questions_text, response_text)

# 5) Output JSON to console
import json
print(json.dumps(results, indent=2, ensure_ascii=False))

# 6) Generate PDF report
create_pdf_report(results, OUTPUT_PDF)
print(f"✓ Graded report saved to {OUTPUT_PDF}")
