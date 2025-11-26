"""
Updated exam grader that uses the multi-agent orchestrator.
This replaces the single-LLM approach with true multi-agent orchestration.
"""

import os
import sys
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import pdfplumber

# Import the orchestrator
from agent_orchestrator import orchestrate_grading

load_dotenv()

# ========== PDF EXTRACTION (reused from original) ==========

def extract_pdf_to_markdown(pdf_path: str) -> str:
    """Extract text from PDF and convert to markdown format."""
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

    def convert_table_to_markdown(table):
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

# ========== GRADING FUNCTION (uses orchestrator) ==========

def grade_exam(
    rubric: str,
    questions: str,
    responses: str,
    exam_type: str = "narrative",
    enable_triage: bool = True
) -> Dict[str, Any]:
    """
    Grade exam using multi-agent orchestration.
    
    Args:
        rubric: Rubric text (optional)
        questions: Exam questions text
        responses: Student responses text
        exam_type: Manual override ("technical", "narrative", or None for auto-triage)
        enable_triage: Whether to use triage agent (default: True)
    
    Returns:
        Grading result with orchestration metadata
    """
    # Use orchestrator instead of direct LLM call
    result = orchestrate_grading(
        exam_text=questions,
        student_response=responses,
        rubric_text=rubric,
        exam_type_override=exam_type if exam_type else None,
        enable_triage=enable_triage
    )
    
    # Extract just the grading result for backward compatibility
    if "error" in result:
        return result
    
    grading_result = result.get("grading_result", {})
    
    # If grading failed, try handoff
    if "error" in grading_result:
        from agent_orchestrator import handle_agent_failure
        exam_type_used = result.get("orchestration_metadata", {}).get("agent_used", exam_type)
        handoff_result = handle_agent_failure(
            failed_exam_type=exam_type_used,
            exam_text=questions,
            student_response=responses,
            rubric_text=rubric,
            error=grading_result.get("error", "Unknown error")
        )
        if "error" not in handoff_result:
            return handoff_result
    
    # Return grading result (can include orchestration metadata if needed)
    return grading_result

