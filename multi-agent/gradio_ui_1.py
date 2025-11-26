import gradio as gr
import os
import sys
import json
from dotenv import load_dotenv
from fpdf import FPDF

# Import multi-agent orchestrator
from agent_orchestrator import extract_pdf_to_markdown

# Add parent directory to path to import vc_grader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vc-pitch-agent'))
from vc_grader import grade_pitch

load_dotenv()

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not found in environment. Please set it in .env file.")

# Utility function to extract text from various file formats
def extract_text_from_file(file_obj):
    if file_obj is None:
        raise ValueError("Student response file is required")
    
    # Handle different Gradio file object structures
    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    
    if file_path.endswith(".pdf"):
        return extract_pdf_to_markdown(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        if hasattr(file_obj, 'read'):
            return file_obj.read().decode("utf-8")
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    else:
        return "Unsupported file format"

# Utility to export JSON result to PDF
def json_to_pdf(json_obj, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=10)
    for line in json.dumps(json_obj, indent=2, ensure_ascii=False).splitlines():
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)

# ========== GRADER HANDLERS ==========
def handle_exam(pdf_path, rubric_path, student_response_file, exam_type, enable_auto_triage):
    try:
        # Validate required inputs
        if pdf_path is None:
            return json.dumps({"error": "Exam PDF is required. Please upload an exam file."}, indent=2), None, None, ""
        
        if student_response_file is None:
            return json.dumps({"error": "Student response file is required. Please upload a student response file."}, indent=2), None, None, ""
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            return json.dumps({"error": "OPENAI_API_KEY not found. Please set it in your .env file in the project root."}, indent=2), None, None, ""
        
        # Extract file paths (handle different Gradio file object structures)
        exam_file_path = pdf_path.name if hasattr(pdf_path, 'name') else str(pdf_path)
        rubric_file_path = rubric_path.name if (rubric_path and hasattr(rubric_path, 'name')) else (str(rubric_path) if rubric_path else None)
        
        # Extract text from files
        try:
            questions_md = extract_pdf_to_markdown(exam_file_path)
        except Exception as e:
            return json.dumps({"error": f"Failed to extract text from exam PDF: {str(e)}"}, indent=2), None, None, ""
        
        try:
            rubric_md = extract_pdf_to_markdown(rubric_file_path) if rubric_file_path else ""
        except Exception as e:
            print(f"Warning: Could not extract rubric (optional): {str(e)}")
            rubric_md = ""
        
        try:
            student_response_md = extract_text_from_file(student_response_file)
        except Exception as e:
            return json.dumps({"error": f"Failed to extract text from student response: {str(e)}"}, indent=2), None, None, ""

        # Grade the exam using multi-agent orchestration
        try:
            from agent_orchestrator import orchestrate_grading
            
            # Determine exam_type_override based on user input
            exam_type_override = None if enable_auto_triage else (exam_type if exam_type else None)
            
            # Use orchestrator directly to get full metadata
            orchestration_result = orchestrate_grading(
                exam_text=questions_md,
                student_response=student_response_md,
                rubric_text=rubric_md,
                exam_type_override=exam_type_override,
                enable_triage=enable_auto_triage
            )
            
            # Extract metadata and grading result
            metadata = orchestration_result.get("orchestration_metadata", {})
            grading_result = orchestration_result.get("grading_result", {})
            
            # Handle errors
            if "error" in orchestration_result:
                metadata_text = format_metadata(metadata)
                return json.dumps(orchestration_result, indent=2, ensure_ascii=False), None, None, metadata_text
            
            if "error" in grading_result:
                # Try handoff
                from agent_orchestrator import handle_agent_failure
                exam_type_used = metadata.get("agent_used", exam_type or "narrative")
                handoff_result = handle_agent_failure(
                    failed_exam_type=exam_type_used,
                    exam_text=questions_md,
                    student_response=student_response_md,
                    rubric_text=rubric_md,
                    error=grading_result.get("error", "Unknown error")
                )
                if "error" not in handoff_result:
                    grading_result = handoff_result
                    metadata["handoff"] = handoff_result.get("handoff_metadata", {})
                else:
                    metadata_text = format_metadata(metadata)
                    return json.dumps(grading_result, indent=2, ensure_ascii=False), None, None, metadata_text
            
            # Format metadata for display
            metadata_text = format_metadata(metadata, grading_result)
            
            # Combine grading result with metadata for JSON output
            output_result = {
                "grading": grading_result,
                "orchestration": metadata
            }
            
            # Save results to JSON and PDF (only for download, not auto-saved to folder)
            agent_used = metadata.get("agent_used", exam_type or "narrative")
            base_name = f"{agent_used}_grade_output"
            json_path = base_name + ".json"
            pdf_output_path = base_name + ".pdf"

            # Create files only for Gradio download (temporary)
            import tempfile
            temp_dir = tempfile.gettempdir()
            json_path = os.path.join(temp_dir, json_path)
            pdf_output_path = os.path.join(temp_dir, pdf_output_path)

            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(output_result, f, indent=2, ensure_ascii=False)
                json_to_pdf(grading_result, pdf_output_path)
            except Exception as e:
                print(f"Warning: Could not save output files: {str(e)}")
                json_path = None
                pdf_output_path = None

            return json.dumps(grading_result, indent=2, ensure_ascii=False), json_path, pdf_output_path, metadata_text
        
        except Exception as e:
            error_msg = f"Orchestration error: {str(e)}"
            print(f"Error in orchestration: {error_msg}")
            import traceback
            traceback.print_exc()
            return json.dumps({"error": error_msg}, indent=2), None, None, ""
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"Error in handle_exam: {error_msg}")
        import traceback
        traceback.print_exc()
        return json.dumps({"error": error_msg}, indent=2), None, None, ""

def format_metadata(metadata: dict, grading_result: dict = None) -> str:
    """Format orchestration metadata for display."""
    lines = ["=== Multi-Agent Orchestration Metadata ===\n"]
    
    # Triage information
    triage = metadata.get("triage", {})
    if triage:
        lines.append(f"🤖 Triage Agent:")
        lines.append(f"   Exam Type: {triage.get('exam_type', 'N/A')}")
        lines.append(f"   Confidence: {triage.get('confidence', 0):.2%}")
        lines.append(f"   Reasoning: {triage.get('reasoning', 'N/A')}")
        lines.append("")
    
    # Agent used
    agent_used = metadata.get("agent_used", "N/A")
    lines.append(f"🎯 Specialist Agent Used: {agent_used.upper()}")
    lines.append("")
    
    # Quality checks
    quality = metadata.get("quality_checks", {})
    if quality:
        lines.append("✅ Quality Checks:")
        lines.append(f"   Exam Length: {quality.get('exam_length', 0):,} characters")
        lines.append(f"   Response Length: {quality.get('response_length', 0):,} characters")
        warnings = quality.get("warnings", [])
        if warnings:
            lines.append("   ⚠️ Warnings:")
            for warning in warnings:
                lines.append(f"      - {warning}")
        else:
            lines.append("   ✓ No warnings")
        lines.append("")
    
    # Handoff information
    handoff = metadata.get("handoff", {})
    if handoff:
        lines.append("🔄 Handoff Information:")
        lines.append(f"   Original Agent: {handoff.get('original_agent', 'N/A')}")
        lines.append(f"   Fallback Agent: {handoff.get('fallback_agent', 'N/A')}")
        if handoff.get("original_error"):
            lines.append(f"   Reason: {handoff.get('original_error')}")
        lines.append("")
    
    # Status
    status = metadata.get("status", "unknown")
    status_emoji = "✅" if status == "success" else "❌"
    lines.append(f"{status_emoji} Status: {status.upper()}")
    
    return "\n".join(lines)

def handle_vc_pitch(audio_file):
    """Handle VC pitch grading using the standalone vc_grader module."""
    try:
        if audio_file is None:
            return json.dumps({"error": "Audio file is required. Please upload an MP3 file."}, indent=2), None, None
        
        # Use the grade_pitch function from vc-pitch-agent
        result = grade_pitch(audio_file)
        
        # Check for errors in result
        if "error" in result:
            return json.dumps(result, indent=2), None, None
        
        # Save results to JSON and PDF (only for download, not auto-saved to folder)
        import tempfile
        temp_dir = tempfile.gettempdir()
        base_name = "vc_pitch_grade_output"
        json_path = os.path.join(temp_dir, base_name + ".json")
        pdf_path = os.path.join(temp_dir, base_name + ".pdf")

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            json_to_pdf(result, pdf_path)
        except Exception as e:
            print(f"Warning: Could not save output files: {str(e)}")
            json_path = None
            pdf_path = None
        
        return json.dumps(result, indent=2, ensure_ascii=False), json_path, pdf_path

    except Exception as e:
        return json.dumps({"error": f"Failed to grade VC pitch: {str(e)}"}, indent=2), None, None

# ========== INTERFACES ==========
with gr.Blocks(title="Multi-Agent Exam Grader") as demo:
    gr.Markdown("# 🤖 Multi-Agent Exam Grading System")
    gr.Markdown("This system uses multiple specialized agents with automatic triage, guardrails, and handoff mechanisms.")
    
    with gr.Tabs():
        with gr.Tab("Text-based Exams"):
            with gr.Row():
                with gr.Column():
                    exam_pdf = gr.File(label="📄 Exam PDF", file_types=[".pdf"])
                    rubric_pdf = gr.File(label="📋 Rubric PDF (optional)", file_types=[".pdf"])
                    student_response = gr.File(label="📝 Student Response", file_types=[".txt", ".md", ".pdf"])
                    
                    with gr.Row():
                        exam_type = gr.Radio(
                            ["narrative", "technical"],
                            label="Exam Type",
                            value="narrative",
                            info="Select manually or enable auto-detection below"
                        )
                        enable_auto_triage = gr.Checkbox(
                            label="🤖 Enable Auto-Triage",
                            value=False,
                            info="Automatically detect exam type using AI triage agent"
                        )
                    
                    submit_btn = gr.Button("🚀 Grade Exam", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")
                
                with gr.Column():
                    metadata_output = gr.Textbox(
                        label="📊 Orchestration Metadata",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        info="Shows which agent was used, confidence scores, and quality checks"
                    )
                    evaluation_output = gr.Textbox(
                        label="📋 Evaluation Output",
                        lines=25,
                        max_lines=50,
                        interactive=False
                    )
                    
                    with gr.Row():
                        json_download = gr.File(label="📥 Download JSON")
                        pdf_download = gr.File(label="📥 Download PDF")
            
            submit_btn.click(
                fn=handle_exam,
                inputs=[exam_pdf, rubric_pdf, student_response, exam_type, enable_auto_triage],
                outputs=[evaluation_output, json_download, pdf_download, metadata_output]
            )
            def clear_all():
                return None, None, None, "narrative", False, "", "", None, None
            
            clear_btn.click(
                fn=clear_all,
                outputs=[exam_pdf, rubric_pdf, student_response, exam_type, enable_auto_triage, 
                        evaluation_output, metadata_output, json_download, pdf_download]
            )
        
        with gr.Tab("VC Pitch Grading"):
            vc_audio = gr.Audio(label="🎤 Upload VC Pitch (MP3)", type="filepath")
            vc_submit = gr.Button("🚀 Grade Pitch", variant="primary")
            vc_output = gr.Textbox(label="📋 VC Pitch Evaluation Output", lines=25, max_lines=50)
            vc_json = gr.File(label="📥 Download JSON")
            vc_pdf = gr.File(label="📥 Download PDF")
            
            vc_submit.click(
                fn=handle_vc_pitch,
                inputs=[vc_audio],
                outputs=[vc_output, vc_json, vc_pdf]
            )
    
    demo.launch(share=True)
