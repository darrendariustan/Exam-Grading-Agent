import gradio as gr
import os
import sys
import json
from exam_grader_agents_multi_1 import extract_pdf_to_markdown, grade_exam
from dotenv import load_dotenv
from fpdf import FPDF

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
def handle_exam(pdf_path, rubric_path, student_response_file, exam_type):
    try:
        # Validate required inputs
        if pdf_path is None:
            return json.dumps({"error": "Exam PDF is required. Please upload an exam file."}, indent=2), None, None
        
        if student_response_file is None:
            return json.dumps({"error": "Student response file is required. Please upload a student response file."}, indent=2), None, None
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            return json.dumps({"error": "OPENAI_API_KEY not found. Please set it in your .env file in the project root."}, indent=2), None, None
        
        # Extract file paths (handle different Gradio file object structures)
        exam_file_path = pdf_path.name if hasattr(pdf_path, 'name') else str(pdf_path)
        rubric_file_path = rubric_path.name if (rubric_path and hasattr(rubric_path, 'name')) else (str(rubric_path) if rubric_path else None)
        
        # Extract text from files
        try:
            questions_md = extract_pdf_to_markdown(exam_file_path)
        except Exception as e:
            return json.dumps({"error": f"Failed to extract text from exam PDF: {str(e)}"}, indent=2), None, None
        
        try:
            rubric_md = extract_pdf_to_markdown(rubric_file_path) if rubric_file_path else ""
        except Exception as e:
            print(f"Warning: Could not extract rubric (optional): {str(e)}")
            rubric_md = ""
        
        try:
            student_response_md = extract_text_from_file(student_response_file)
        except Exception as e:
            return json.dumps({"error": f"Failed to extract text from student response: {str(e)}"}, indent=2), None, None

        # Grade the exam
        try:
            result = grade_exam(rubric_md, questions_md, student_response_md, exam_type=exam_type)
        except Exception as e:
            return json.dumps({"error": f"Failed to grade exam: {str(e)}. Please check your API key and try again."}, indent=2), None, None

        # Check if result contains an error
        if isinstance(result, dict) and "error" in result:
            return json.dumps(result, indent=2, ensure_ascii=False), None, None

        # Save results to JSON and PDF
        base_name = f"{exam_type}_grade_output"
        json_path = base_name + ".json"
        pdf_output_path = base_name + ".pdf"

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            json_to_pdf(result, pdf_output_path)
        except Exception as e:
            print(f"Warning: Could not save output files: {str(e)}")

        return json.dumps(result, indent=2, ensure_ascii=False), json_path, pdf_output_path
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"Error in handle_exam: {error_msg}")
        import traceback
        traceback.print_exc()
        return json.dumps({"error": error_msg}, indent=2), None, None

def handle_vc_pitch(audio_file):
    """Handle VC pitch grading using the standalone vc_grader module."""
    try:
        # Use the grade_pitch function from vc-pitch-agent
        result = grade_pitch(audio_file)
        
        # Check for errors in result
        if "error" in result:
            return json.dumps(result, indent=2), None, None
        
        # Save results to JSON and PDF
        base_name = "vc_pitch_grade_output"
        json_path = base_name + ".json"
        pdf_path = base_name + ".pdf"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        json_to_pdf(result, pdf_path)
        
        return json.dumps(result, indent=2, ensure_ascii=False), json_path, pdf_path

    except Exception as e:
        return f"Error: {str(e)}", None, None

# ========== INTERFACES ==========
exam_tab = gr.Interface(
    fn=lambda pdf, rubric, student, exam_type: handle_exam(pdf, rubric, student, exam_type),
    inputs=[
        gr.File(label="Exam PDF"),
        gr.File(label="Rubric PDF (optional)"),
        gr.File(label="Student Response (.txt, .md, .pdf)"),
        gr.Radio(["narrative", "technical"], label="Exam Type")
    ],
    outputs=[
        gr.Textbox(label="Evaluation Output"),
        gr.File(label="Download JSON"),
        gr.File(label="Download PDF")
    ],
    title="Narrative & Technical Exam Grader"
)

vc_tab = gr.Interface(
    fn=handle_vc_pitch,
    inputs=[gr.Audio(label="Upload VC Pitch (MP3)", type="filepath")],
    outputs=[
        gr.Textbox(label="VC Pitch Evaluation Output"),
        gr.File(label="Download JSON"),
        gr.File(label="Download PDF")
    ],
    title="VC Pitch Grader"
)

gr.TabbedInterface([exam_tab, vc_tab], ["Text-based Exams", "VC Pitch Grading"]).launch(share=True)