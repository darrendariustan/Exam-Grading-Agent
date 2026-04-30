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

# Add paths to import specialist agents (use absolute paths)
_base_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_base_dir)
_technical_agent_path = os.path.join(_parent_dir, 'technical-agent')
_narrative_agent_path = os.path.join(_parent_dir, 'narrative-agent')
_vc_pitch_agent_path = os.path.join(_parent_dir, 'vc-pitch-agent')

sys.path.insert(0, _technical_agent_path)
sys.path.insert(0, _narrative_agent_path)
sys.path.insert(0, _vc_pitch_agent_path)

# Import specialist agents
try:
    from tech_grading_agent import grade_exam as grade_technical_exam
except ImportError as e:
    print(f"Warning: Could not import technical agent: {e}")
    grade_technical_exam = None
except Exception as e:
    print(f"Warning: Error importing technical agent: {e}")
    import traceback
    traceback.print_exc()
    grade_technical_exam = None

try:
    from exam_grader_agents import grade_exam as grade_narrative_exam
except ImportError as e:
    print(f"Warning: Could not import narrative agent from {_narrative_agent_path}: {e}")
    print(f"  Narrative agent path exists: {os.path.exists(_narrative_agent_path)}")
    if os.path.exists(_narrative_agent_path):
        print(f"  Files in narrative-agent: {os.listdir(_narrative_agent_path)}")
    grade_narrative_exam = None
except Exception as e:
    print(f"Warning: Error importing narrative agent: {e}")
    import traceback
    traceback.print_exc()
    grade_narrative_exam = None

try:
    # Try new agent version first, fallback to old
    from vc_grader_agent import grade_pitch_agent as grade_pitch
except ImportError:
    try:
        from vc_grader import grade_pitch
    except ImportError:
        grade_pitch = None

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def is_mock_mode_enabled() -> bool:
    """Enable deterministic local testing without external API calls."""
    return os.getenv("MOCK_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

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

TRIAGE_PROMPT = """You are a triage agent that classifies exam types and input formats. Analyze the input and determine the most appropriate grading agent.

Input types:
- "text": Written exam questions and responses (PDF, TXT, MD files)
- "audio": Audio/video pitch presentations (MP3, WAV, MP4 files)

Exam types (for text inputs):
- "technical": Factual knowledge, mathematical reasoning, coding, technical problem-solving
- "narrative": Open-ended responses, strategic thinking, reflective writing, business cases, essays
- "vc_pitch": Audio/video pitches, presentations, entrepreneurial pitches (for audio inputs)

Consider:
- Audio/video files → input_type: "audio", exam_type: "vc_pitch"
- Text files with mathematical formulas, code, technical diagrams → input_type: "text", exam_type: "technical"
- Text files with essay questions, case studies, strategic analysis → input_type: "text", exam_type: "narrative"

Respond ONLY with JSON:
{
  "input_type": "text" | "audio",
  "exam_type": "technical" | "narrative" | "vc_pitch",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
"""

def detect_input_type(file_path: str = None, audio_file: str = None) -> str:
    """
    Detect input type from file extensions.
    
    Args:
        file_path: Path to text file (PDF, TXT, MD)
        audio_file: Path to audio file (MP3, WAV, etc.)
        
    Returns:
        "text" or "audio"
    """
    if audio_file:
        return "audio"
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".mp3", ".wav", ".mp4", ".m4a", ".ogg"]:
            return "audio"
        return "text"
    return "text"

def triage_exam_type(questions_text: str = "", rubric_text: str = "", input_type: str = "text") -> Dict[str, Any]:
    """
    Triage agent: Automatically classifies exam type and input format.
    
    Args:
        questions_text: Extracted exam questions (for text inputs)
        rubric_text: Optional rubric text
        input_type: "text" or "audio" (detected from file)
        
    Returns:
        Dict with input_type, exam_type, confidence, reasoning
    """
    try:
        if input_type == "audio":
            # Audio inputs are always VC pitch
            return {
                "input_type": "audio",
                "exam_type": "vc_pitch",
                "confidence": 1.0,
                "reasoning": "Audio file detected - routing to VC pitch agent"
            }

        if is_mock_mode_enabled():
            combined = f"{questions_text}\n{rubric_text}".lower()
            technical_hints = ["code", "sql", "equation", "calculate", "algorithm", "debug"]
            is_technical = any(token in combined for token in technical_hints)
            return {
                "input_type": "text",
                "exam_type": "technical" if is_technical else "narrative",
                "confidence": 0.85,
                "reasoning": "MOCK_MODE enabled - heuristic triage used"
            }
        
        # Text input - analyze content
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
        result["input_type"] = "text"  # Ensure consistency
        return result
    except Exception as e:
        # Fallback to narrative if triage fails
        return {
            "input_type": "text",
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
            # Handle None return (parsing error)
            if result is None:
                return {"error": "Technical agent failed to parse response"}
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
    exam_text: str = "",
    student_response: str = "",
    rubric_text: str = "",
    exam_type_override: Optional[str] = None,
    enable_triage: bool = True,
    audio_file: Optional[str] = None,
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
        # Step 0: Detect input type
        input_type = detect_input_type(
            file_path=kwargs.get("exam_file_path"),
            audio_file=audio_file
        )
        result["orchestration_metadata"]["mock_mode"] = is_mock_mode_enabled()
        
        # Step 1: Guardrails - Input validation
        if input_type == "audio":
            if not audio_file:
                return {"error": "Audio file is required for VC pitch grading"}
            # Skip text validation for audio
        else:
            is_valid, error_msg = validate_inputs(exam_text, student_response, 
                                                 exam_type_override or "narrative")
            if not is_valid:
                return {"error": f"Guardrail validation failed: {error_msg}"}
        
        # Step 2: Quality checks (only for text inputs)
        if input_type == "text":
            quality = check_content_quality(exam_text, student_response)
            result["orchestration_metadata"]["quality_checks"] = quality
        else:
            result["orchestration_metadata"]["quality_checks"] = {
                "input_type": "audio",
                "audio_file": audio_file
            }
        
        # Step 3: Triage (if enabled and no override)
        if enable_triage and not exam_type_override:
            triage_result = triage_exam_type(exam_text, rubric_text, input_type)
            exam_type = triage_result.get("exam_type", "narrative")
            result["orchestration_metadata"]["triage"] = triage_result
        else:
            exam_type = exam_type_override or ("vc_pitch" if input_type == "audio" else "narrative")
            result["orchestration_metadata"]["triage"] = {
                "input_type": input_type,
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
            mp3_path=audio_file,
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

