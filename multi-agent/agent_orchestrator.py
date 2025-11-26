"""
Multi-Agent Orchestration System for Exam Grading

This module implements a true multi-agent system with:
- Triage Agent: Automatically classifies exam type
- Guardrails: Input validation and safety checks
- Specialist Agents: Routes to technical, narrative, or VC-pitch agents
- Handoffs: Error recovery and fallback mechanisms
"""

import os
import sys
import json
import openai
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Add paths to import specialist agents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'technical-agent'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'narrative-agent'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vc-pitch-agent'))

# Import specialist agents
try:
    from tech_grading_agent import grade_exam as grade_technical_exam
except ImportError:
    grade_technical_exam = None

try:
    from exam_grader_agents import grade_exam as grade_narrative_exam
except ImportError:
    grade_narrative_exam = None

try:
    from vc_grader import grade_pitch
except ImportError:
    grade_pitch = None

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ========== PDF EXTRACTION UTILITY ==========

def extract_pdf_to_markdown(pdf_path: str) -> str:
    """Extract text from PDF and convert to markdown format."""
    import pdfplumber
    
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

# ========== TRIAGE AGENT ==========

TRIAGE_PROMPT = """You are a triage agent that classifies exam types. Analyze the exam content and determine the most appropriate grading agent.

Exam types:
- "technical": Factual knowledge, mathematical reasoning, coding, technical problem-solving
- "narrative": Open-ended responses, strategic thinking, reflective writing, business cases, essays
- "vc_pitch": Audio/video pitches, presentations, entrepreneurial pitches

Consider:
- Presence of mathematical formulas, code, technical diagrams → technical
- Presence of essay questions, case studies, strategic analysis → narrative
- Audio/video files → vc_pitch

Respond ONLY with JSON:
{
  "exam_type": "technical" | "narrative" | "vc_pitch",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
"""

def triage_exam_type(questions_text: str, rubric_text: str = "") -> Dict[str, Any]:
    """
    Triage agent: Automatically classifies exam type.
    
    Args:
        questions_text: Extracted exam questions
        rubric_text: Optional rubric text
        
    Returns:
        Dict with exam_type, confidence, reasoning
    """
    try:
        combined_text = f"Questions:\n{questions_text}\n\n"
        if rubric_text:
            combined_text += f"Rubric:\n{rubric_text}\n"
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": TRIAGE_PROMPT},
                {"role": "user", "content": combined_text}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        # Fallback to narrative if triage fails
        return {
            "exam_type": "narrative",
            "confidence": 0.5,
            "reasoning": f"Triage failed: {str(e)}, defaulting to narrative"
        }

# ========== GUARDRAILS ==========

def validate_inputs(exam_text: str, student_response: str, exam_type: str) -> Tuple[bool, Optional[str]]:
    """
    Guardrails: Validate inputs before processing.
    
    Returns:
        (is_valid, error_message)
    """
    if not exam_text or len(exam_text.strip()) < 10:
        return False, "Exam content is too short or empty"
    
    if not student_response or len(student_response.strip()) < 5:
        return False, "Student response is too short or empty"
    
    if exam_type not in ["technical", "narrative", "vc_pitch"]:
        return False, f"Invalid exam type: {exam_type}"
    
    # Check for suspicious content (basic safety)
    suspicious_keywords = ["<script>", "javascript:", "eval("]
    if any(keyword in student_response.lower() for keyword in suspicious_keywords):
        return False, "Student response contains potentially unsafe content"
    
    return True, None

def check_content_quality(exam_text: str, student_response: str) -> Dict[str, Any]:
    """
    Quality checks for content before grading.
    
    Returns:
        Dict with quality metrics and warnings
    """
    quality = {
        "exam_length": len(exam_text),
        "response_length": len(student_response),
        "warnings": []
    }
    
    if len(exam_text) < 50:
        quality["warnings"].append("Exam content seems very short")
    
    if len(student_response) < 20:
        quality["warnings"].append("Student response seems very short")
    
    if len(student_response) > 50000:
        quality["warnings"].append("Student response is very long, may take longer to process")
    
    return quality

# ========== AGENT ROUTER ==========

def route_to_agent(
    exam_type: str,
    questions: str,
    responses: str,
    rubric: str = "",
    **kwargs
) -> Dict[str, Any]:
    """
    Routes to appropriate specialist agent based on exam type.
    
    Args:
        exam_type: "technical", "narrative", or "vc_pitch"
        questions: Exam questions text
        responses: Student responses text
        rubric: Optional rubric text
        **kwargs: Additional arguments (e.g., mp3_path for VC pitch)
    
    Returns:
        Grading result from specialist agent
    """
    try:
        if exam_type == "technical":
            if grade_technical_exam is None:
                return {"error": "Technical agent not available (import failed)"}
            # Technical agent signature: grade_exam(questions_markdown, answers_text, rubric_markdown=None)
            result = grade_technical_exam(
                questions_markdown=questions,
                answers_text=responses,
                rubric_markdown=rubric if rubric else None
            )
            # Normalize output format
            if result and ("question_1" in result or "error" in result):
                return result
            else:
                return {"error": "Technical agent returned unexpected format", "raw": result}
        
        elif exam_type == "narrative":
            if grade_narrative_exam is None:
                return {"error": "Narrative agent not available (import failed)"}
            # Narrative agent signature: grade_exam(rubric: str, questions: str, responses: str)
            result = grade_narrative_exam(
                rubric=rubric,
                questions=questions,
                responses=responses
            )
            # Narrative agent already returns structured format
            return result
        
        elif exam_type == "vc_pitch":
            if grade_pitch is None:
                return {"error": "VC pitch agent not available (import failed)"}
            # VC pitch agent signature: grade_pitch(mp3_path: str)
            mp3_path = kwargs.get("mp3_path")
            if not mp3_path:
                return {"error": "MP3 file path required for VC pitch grading"}
            result = grade_pitch(mp3_path)
            return result
        
        else:
            return {"error": f"Unknown exam type: {exam_type}"}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"Agent routing failed: {str(e)}",
            "exam_type": exam_type
        }

# ========== MAIN ORCHESTRATOR ==========

def orchestrate_grading(
    exam_text: str,
    student_response: str,
    rubric_text: str = "",
    exam_type_override: Optional[str] = None,
    enable_triage: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Main orchestration function that coordinates all agents.
    
    Args:
        exam_text: Exam questions text
        student_response: Student response text
        rubric_text: Optional rubric text
        exam_type_override: Manual override for exam type (bypasses triage)
        enable_triage: Whether to use triage agent (default: True)
        **kwargs: Additional arguments for specific agents
    
    Returns:
        Dict with grading result and metadata
    """
    result = {
        "orchestration_metadata": {},
        "grading_result": {}
    }
    
    try:
        # Step 1: Guardrails - Input validation
        is_valid, error_msg = validate_inputs(exam_text, student_response, 
                                             exam_type_override or "narrative")
        if not is_valid:
            return {"error": f"Guardrail validation failed: {error_msg}"}
        
        # Step 2: Quality checks
        quality = check_content_quality(exam_text, student_response)
        result["orchestration_metadata"]["quality_checks"] = quality
        
        # Step 3: Triage (if enabled and no override)
        if enable_triage and not exam_type_override:
            triage_result = triage_exam_type(exam_text, rubric_text)
            exam_type = triage_result.get("exam_type", "narrative")
            result["orchestration_metadata"]["triage"] = triage_result
        else:
            exam_type = exam_type_override or "narrative"
            result["orchestration_metadata"]["triage"] = {
                "exam_type": exam_type,
                "confidence": 1.0,
                "reasoning": "Manual override or triage disabled"
            }
        
        # Step 4: Route to specialist agent
        grading_result = route_to_agent(
            exam_type=exam_type,
            questions=exam_text,
            responses=student_response,
            rubric=rubric_text,
            **kwargs
        )
        
        result["grading_result"] = grading_result
        result["orchestration_metadata"]["agent_used"] = exam_type
        result["orchestration_metadata"]["status"] = "success"
        
        return result
    
    except Exception as e:
        return {
            "error": f"Orchestration failed: {str(e)}",
            "orchestration_metadata": {
                "status": "error",
                "error_type": type(e).__name__
            }
        }

# ========== HANDOFF MECHANISM ==========

def handle_agent_failure(
    failed_exam_type: str,
    exam_text: str,
    student_response: str,
    rubric_text: str = "",
    error: str = ""
) -> Dict[str, Any]:
    """
    Handoff mechanism: If one agent fails, try fallback agent.
    
    Args:
        failed_exam_type: The exam type that failed
        exam_text: Exam questions
        student_response: Student response
        rubric_text: Optional rubric
        error: Error message from failed agent
    
    Returns:
        Result from fallback agent or error
    """
    # Fallback strategy: technical -> narrative -> error
    fallback_map = {
        "technical": "narrative",
        "narrative": "technical",
        "vc_pitch": "narrative"  # Can't fallback VC pitch to text-based
    }
    
    fallback_type = fallback_map.get(failed_exam_type)
    if not fallback_type:
        return {"error": f"No fallback available for {failed_exam_type}"}
    
    try:
        result = route_to_agent(
            exam_type=fallback_type,
            questions=exam_text,
            responses=student_response,
            rubric=rubric_text
        )
        result["handoff_metadata"] = {
            "original_agent": failed_exam_type,
            "fallback_agent": fallback_type,
            "original_error": error
        }
        return result
    except Exception as e:
        return {
            "error": f"Fallback agent also failed: {str(e)}",
            "handoff_metadata": {
                "original_agent": failed_exam_type,
                "fallback_agent": fallback_type,
                "original_error": error
            }
        }

