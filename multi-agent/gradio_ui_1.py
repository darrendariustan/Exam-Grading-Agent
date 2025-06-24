import gradio as gr
import os
import json
from exam_grader_agents_multi_1 import extract_pdf_to_markdown, analyze_audio, grade_exam
from dotenv import load_dotenv
from fpdf import FPDF
import openai

load_dotenv()

# Utility function to extract text from various file formats
def extract_text_from_file(file_obj):
    if file_obj.name.endswith(".pdf"):
        return extract_pdf_to_markdown(file_obj.name)
    elif file_obj.name.endswith(".txt") or file_obj.name.endswith(".md"):
        return file_obj.read().decode("utf-8")
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
    questions_md = extract_pdf_to_markdown(pdf_path.name)
    rubric_md = extract_pdf_to_markdown(rubric_path.name) if rubric_path else ""
    student_response_md = extract_text_from_file(student_response_file)

    result = grade_exam(rubric_md, questions_md, student_response_md, exam_type=exam_type)

    # Save results to JSON and PDF
    base_name = f"{exam_type}_grade_output"
    json_path = base_name + ".json"
    pdf_path = base_name + ".pdf"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    json_to_pdf(result, pdf_path)

    return json.dumps(result, indent=2, ensure_ascii=False), json_path, pdf_path

def handle_vc_pitch(audio_file):
    audio_metrics = analyze_audio(audio_file)
    transcript = audio_metrics["transcript"]
    duration_minutes = audio_metrics["duration"] / 60

    prompt = f"""
Pitch transcript:
{transcript}

Audio metrics:
• Words-per-minute: {audio_metrics['wpm']:.1f}
• Pause ratio: {audio_metrics['silence_ratio']:.1%}

You are a seasoned VC pitch grader. For a {duration_minutes:.1f}-minute audio pitch, give each dimension a score from 1 (poor) to 10 (excellent), using the following anchors:

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
{{
  "Problem": <1–10>,
  "Market": <1–10>,
  "Solution": <1–10>,
  "Delivery": <1–10>,
  "Feedback": "<one sentence actionable feedback for each anchor>"
}}
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": "You are a helpful pitch grader."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        result = response.choices[0].message.content

        base_name = "vc_pitch_grade_output"
        json_path = base_name + ".json"
        pdf_path = base_name + ".pdf"

        try:
            json_result = json.loads(result)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            json_to_pdf(json_result, pdf_path)
            return result, json_path, pdf_path
        except json.JSONDecodeError:
            return result, None, None

    except openai.OpenAIError as e:
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

gr.TabbedInterface([exam_tab, vc_tab], ["Text-based Exams", "VC Pitch Grading"]).launch()