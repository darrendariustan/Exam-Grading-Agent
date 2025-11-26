"""
Batch Multi-Agent Grading UI
Supports grading multiple students at once with table output and batch downloads
"""

import gradio as gr
import os
import sys
import json
import tempfile
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from fpdf import FPDF

# Import multi-agent orchestrator
from agent_orchestrator import extract_pdf_to_markdown, orchestrate_grading

# Add parent directory to path to import vc_grader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vc-pitch-agent'))
try:
    from vc_grader_agent import grade_pitch_agent
except ImportError:
    from vc_grader import grade_pitch as grade_pitch_agent

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not found in environment. Please set it in .env file.")

def extract_text_from_file(file_obj):
    """Extract text from a file object (PDF, TXT, MD)."""
    if file_obj is None:
        raise ValueError("File is required")
    
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

def json_to_pdf(json_obj, output_path):
    """Convert JSON result to PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=10)
    for line in json.dumps(json_obj, indent=2, ensure_ascii=False).splitlines():
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)

def detect_file_type(file_obj):
    """Detect if file is audio or text-based."""
    if file_obj is None:
        return None
    
    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in [".mp3", ".wav", ".mp4", ".m4a", ".ogg"]:
        return "audio"
    elif ext in [".pdf", ".txt", ".md"]:
        return "text"
    return "unknown"

def split_combined_file(content: str) -> tuple:
    """Try to split a file that might contain both questions and answers."""
    import re
    
    question_patterns = [
        r'Question\s+\d+[:.]',
        r'Q\d+[:.]',
        r'Problem\s+\d+[:.]',
        r'^\d+[\.\)]\s+',
    ]
    
    has_questions = any(re.search(pattern, content, re.MULTILINE | re.IGNORECASE) for pattern in question_patterns)
    
    answer_patterns = [
        r'Answer\s*[:.]',
        r'Solution\s*[:.]',
        r'Response\s*[:.]',
    ]
    
    has_answers = any(re.search(pattern, content, re.MULTILINE | re.IGNORECASE) for pattern in answer_patterns)
    
    if has_questions and has_answers:
        return content, content
    elif has_questions:
        return content, ""
    else:
        return "", content

def grade_single_student(
    student_file,
    rubric_content: str,
    exam_questions: str = None
) -> Dict[str, Any]:
    """
    Grade a single student file.
    
    Returns:
        Dict with keys: student_name, file_type, grading_result, metadata, error
    """
    try:
        # Get student name from filename
        file_path = student_file.name if hasattr(student_file, 'name') else str(student_file)
        student_name = os.path.splitext(os.path.basename(file_path))[0]
        
        file_type = detect_file_type(student_file)
        
        # Process audio input (VC pitch)
        if file_type == "audio":
            try:
                orchestration_result = orchestrate_grading(
                    exam_text="",
                    student_response="",
                    rubric_text="",
                    exam_type_override="vc_pitch",
                    enable_triage=False,
                    audio_file=file_path
                )
                
                metadata = orchestration_result.get("orchestration_metadata", {})
                grading_result = orchestration_result.get("grading_result", {})
                
                if "error" in orchestration_result or "error" in grading_result:
                    return {
                        "student_name": student_name,
                        "file_type": "audio",
                        "grading_result": grading_result,
                        "metadata": metadata,
                        "error": grading_result.get("error", "Unknown error")
                    }
                
                return {
                    "student_name": student_name,
                    "file_type": "audio",
                    "grading_result": grading_result,
                    "metadata": metadata,
                    "error": None
                }
            except Exception as e:
                return {
                    "student_name": student_name,
                    "file_type": "audio",
                    "grading_result": {},
                    "metadata": {},
                    "error": str(e)
                }
        
        # Process text input (exam grading)
        try:
            main_content = extract_text_from_file(student_file)
        except Exception as e:
            return {
                "student_name": student_name,
                "file_type": "text",
                "grading_result": {},
                "metadata": {},
                "error": f"Failed to extract text: {str(e)}"
            }
        
        # Determine exam questions and student responses
        questions_from_main, responses_from_main = split_combined_file(main_content)
        
        if exam_questions:
            # Exam questions provided separately (from rubric or first file)
            questions_md = exam_questions
            student_response_md = responses_from_main if responses_from_main else main_content
        elif questions_from_main and responses_from_main:
            # Combined file
            questions_md = questions_from_main
            student_response_md = responses_from_main
        elif questions_from_main:
            # Questions only - need to use rubric or error
            if rubric_content:
                questions_md = rubric_content
                student_response_md = main_content
            else:
                return {
                    "student_name": student_name,
                    "file_type": "text",
                    "grading_result": {},
                    "metadata": {},
                    "error": "File contains questions only. Please provide exam questions in rubric field."
                }
        else:
            # Student response only
            if rubric_content:
                questions_md = rubric_content
                student_response_md = main_content
            else:
                return {
                    "student_name": student_name,
                    "file_type": "text",
                    "grading_result": {},
                    "metadata": {},
                    "error": "Student response detected but no exam questions found. Please provide exam questions in rubric field."
                }
        
        # Use orchestrator
        orchestration_result = orchestrate_grading(
            exam_text=questions_md,
            student_response=student_response_md,
            rubric_text=rubric_content if "Rubric" in rubric_content else "",
            exam_type_override=None,
            enable_triage=True,
            exam_file_path=file_path
        )
        
        metadata = orchestration_result.get("orchestration_metadata", {})
        grading_result = orchestration_result.get("grading_result", {})
        
        if "error" in orchestration_result:
            return {
                "student_name": student_name,
                "file_type": "text",
                "grading_result": orchestration_result,
                "metadata": metadata,
                "error": orchestration_result.get("error", "Unknown error")
            }
        
        if "error" in grading_result:
            from agent_orchestrator import handle_agent_failure
            exam_type_used = metadata.get("agent_used", "narrative")
            handoff_result = handle_agent_failure(
                failed_exam_type=exam_type_used,
                exam_text=questions_md,
                student_response=student_response_md,
                rubric_text=rubric_content,
                error=grading_result.get("error", "Unknown error")
            )
            if "error" not in handoff_result:
                grading_result = handoff_result
                metadata["handoff"] = handoff_result.get("handoff_metadata", {})
            else:
                return {
                    "student_name": student_name,
                    "file_type": "text",
                    "grading_result": grading_result,
                    "metadata": metadata,
                    "error": grading_result.get("error", "Unknown error")
                }
        
        return {
            "student_name": student_name,
            "file_type": "text",
            "grading_result": grading_result,
            "metadata": metadata,
            "error": None
        }
    
    except Exception as e:
        return {
            "student_name": student_name if 'student_name' in locals() else "Unknown",
            "file_type": "unknown",
            "grading_result": {},
            "metadata": {},
            "error": f"Unexpected error: {str(e)}"
        }

def extract_exam_questions_from_first_file(files: List) -> Tuple[str, List]:
    """
    Check if first file contains exam questions.
    Returns: (exam_questions, remaining_files)
    """
    if not files or len(files) == 0:
        return None, files
    
    try:
        first_file = files[0]
        content = extract_text_from_file(first_file)
        questions_from_main, responses_from_main = split_combined_file(content)
        
        if questions_from_main and not responses_from_main:
            # First file is exam questions only
            return questions_from_main, files[1:]
        elif questions_from_main and responses_from_main:
            # Combined file - use for both
            return questions_from_main, files
    except:
        pass
    
    return None, files

def format_metadata_for_batch(results: List[Dict[str, Any]]) -> str:
    """Format orchestration metadata for all students in batch."""
    lines = ["=== Multi-Agent Orchestration Summary ===\n"]
    
    # Count agent usage
    agent_counts = {}
    triage_results = []
    handoff_count = 0
    
    for result in results:
        metadata = result.get("metadata", {})
        agent_used = metadata.get("agent_used", "N/A")
        agent_counts[agent_used] = agent_counts.get(agent_used, 0) + 1
        
        triage = metadata.get("triage", {})
        if triage:
            triage_results.append({
                "student": result.get("student_name", "Unknown"),
                "exam_type": triage.get("exam_type", "N/A"),
                "confidence": triage.get("confidence", 0)
            })
        
        if metadata.get("handoff"):
            handoff_count += 1
    
    lines.append("📊 Agent Usage Statistics:")
    for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"   {agent.upper()}: {count} student(s)")
    lines.append("")
    
    if triage_results:
        lines.append("🤖 Triage Agent Results:")
        for triage in triage_results:
            lines.append(f"   {triage['student']}: {triage['exam_type']} (confidence: {triage['confidence']:.1%})")
        lines.append("")
    
    if handoff_count > 0:
        lines.append(f"🔄 Handoffs: {handoff_count} student(s) required fallback to alternative agent")
        lines.append("")
    
    # Show detailed metadata for each student
    lines.append("=== Detailed Metadata by Student ===\n")
    for result in results:
        student_name = result.get("student_name", "Unknown")
        metadata = result.get("metadata", {})
        error = result.get("error")
        
        lines.append(f"📝 {student_name}:")
        
        if error:
            lines.append(f"   ❌ Error: {error}")
        else:
            triage = metadata.get("triage", {})
            if triage:
                input_type = triage.get("input_type", "text")
                lines.append(f"   📥 Input Type: {input_type.upper()}")
                lines.append(f"   🤖 Triage: {triage.get('exam_type', 'N/A')} (confidence: {triage.get('confidence', 0):.1%})")
            
            agent_used = metadata.get("agent_used", "N/A")
            lines.append(f"   🎯 Agent Used: {agent_used.upper()}")
            
            quality = metadata.get("quality_checks", {})
            if quality:
                if quality.get("input_type") == "audio":
                    lines.append(f"   ✅ Audio File: {quality.get('audio_file', 'N/A')}")
                else:
                    lines.append(f"   ✅ Exam: {quality.get('exam_length', 0):,} chars, Response: {quality.get('response_length', 0):,} chars")
                    warnings = quality.get("warnings", [])
                    if warnings:
                        lines.append(f"   ⚠️ Warnings: {len(warnings)}")
            
            handoff = metadata.get("handoff", {})
            if handoff:
                lines.append(f"   🔄 Handoff: {handoff.get('original_agent', 'N/A')} → {handoff.get('fallback_agent', 'N/A')}")
            
            status = metadata.get("status", "unknown")
            status_emoji = "✅" if status == "success" else "❌"
            lines.append(f"   {status_emoji} Status: {status.upper()}")
        
        lines.append("")
    
    return "\n".join(lines)

def handle_batch_grading(student_files: List, rubric_file) -> Tuple[str, str, str, Dict[str, str], Dict[str, str], str]:
    """
    Handle batch grading of multiple students.
    
    Returns:
        (results_table_html, summary_text, metadata_text, individual_json_files, individual_pdf_files, combined_file)
    """
    try:
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            return (
                "<p style='color: red;'>Error: OPENAI_API_KEY not found. Please set it in your .env file.</p>",
                "Error: API key not found",
                {},
                {},
                None
            )
        
        if not student_files or len(student_files) == 0:
            return (
                "<p>Please upload at least one student file to begin grading.</p>",
                "No files uploaded",
                {},
                {},
                None
            )
        
        # Extract rubric content
        rubric_content = ""
        if rubric_file:
            try:
                rubric_file_path = rubric_file.name if hasattr(rubric_file, 'name') else str(rubric_file)
                rubric_content = extract_pdf_to_markdown(rubric_file_path) if rubric_file_path.endswith(".pdf") else extract_text_from_file(rubric_file)
            except Exception as e:
                print(f"Warning: Could not extract rubric: {str(e)}")
        
        # Check if first file contains exam questions
        exam_questions, files_to_grade = extract_exam_questions_from_first_file(student_files)
        
        # Grade each student
        results = []
        temp_dir = tempfile.gettempdir()
        individual_json_files = {}
        individual_pdf_files = {}
        all_grading_results = []
        
        for i, student_file in enumerate(files_to_grade):
            print(f"Grading student {i+1}/{len(files_to_grade)}...")
            result = grade_single_student(student_file, rubric_content, exam_questions)
            results.append(result)
            
            # Create individual download files
            if result.get("error") is None:
                student_name = result["student_name"]
                grading_result = result["grading_result"]
                metadata = result["metadata"]
                
                # Save individual JSON
                json_path = os.path.join(temp_dir, f"{student_name}_grade.json")
                output_result = {"grading": grading_result, "orchestration": metadata}
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(output_result, f, indent=2, ensure_ascii=False)
                individual_json_files[student_name] = json_path
                
                # Save individual PDF
                pdf_path = os.path.join(temp_dir, f"{student_name}_grade.pdf")
                try:
                    json_to_pdf(grading_result, pdf_path)
                    individual_pdf_files[student_name] = pdf_path
                except Exception as e:
                    print(f"Warning: Could not create PDF for {student_name}: {str(e)}")
                
                all_grading_results.append({
                    "student": student_name,
                    "result": output_result
                })
        
        # Create results table
        table_rows = []
        for result in results:
            student_name = result["student_name"]
            file_type = result["file_type"]
            
            if result.get("error"):
                status = "❌ Error"
                score = "N/A"
                agent = "N/A"
                error_msg = result["error"]
            else:
                grading_result = result["grading_result"]
                metadata = result["metadata"]
                
                # Extract score
                if "total_score" in grading_result:
                    score = f"{grading_result['total_score']:.1f}"
                elif "overall_score" in grading_result:
                    score = f"{grading_result['overall_score']:.1f}"
                elif "scores" in grading_result and len(grading_result["scores"]) > 0:
                    scores = [s.get("score", 0) for s in grading_result["scores"] if isinstance(s.get("score"), (int, float))]
                    score = f"{sum(scores) / len(scores):.1f}" if scores else "N/A"
                else:
                    score = "N/A"
                
                agent = metadata.get("agent_used", "N/A").upper()
                status = "✅ Success"
                error_msg = ""
            
            table_rows.append([
                student_name,
                file_type.upper(),
                status,
                score,
                agent,
                error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
            ])
        
        # Generate HTML table
        table_html = """
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
            <thead>
                <tr style="background-color: #2d2d2d; color: white;">
                    <th style="padding: 10px; border: 1px solid #555;">Student</th>
                    <th style="padding: 10px; border: 1px solid #555;">Type</th>
                    <th style="padding: 10px; border: 1px solid #555;">Status</th>
                    <th style="padding: 10px; border: 1px solid #555;">Score</th>
                    <th style="padding: 10px; border: 1px solid #555;">Agent</th>
                    <th style="padding: 10px; border: 1px solid #555;">Error</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for row in table_rows:
            table_html += "<tr>"
            for cell in row:
                table_html += f"<td style='padding: 8px; border: 1px solid #555;'>{cell}</td>"
            table_html += "</tr>"
        
        table_html += """
            </tbody>
        </table>
        """
        
        # Generate summary
        successful = sum(1 for r in results if r.get("error") is None)
        failed = len(results) - successful
        summary = f"Batch Grading Complete: {successful} successful, {failed} failed out of {len(results)} students"
        
        # Format orchestration metadata
        metadata_text = format_metadata_for_batch(results)
        
        # Create combined file
        combined_file = None
        if all_grading_results:
            try:
                combined_json_path = os.path.join(temp_dir, "batch_grading_results.json")
                with open(combined_json_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "batch_summary": {
                            "total_students": len(results),
                            "successful": successful,
                            "failed": failed
                        },
                        "results": all_grading_results
                    }, f, indent=2, ensure_ascii=False)
                combined_file = combined_json_path
            except Exception as e:
                print(f"Warning: Could not create combined file: {str(e)}")
        
        return (
            table_html,
            summary,
            metadata_text,
            individual_json_files,
            individual_pdf_files,
            combined_file
        )
    
    except Exception as e:
        error_msg = f"Unexpected error in batch grading: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return (
            f"<p style='color: red;'>{error_msg}</p>",
            f"Error: {error_msg}",
            f"Error occurred: {error_msg}",
            {},
            {},
            None
        )

# ========== BATCH GRADING INTERFACE ==========
with gr.Blocks(title="Batch Multi-Agent Exam Grader") as demo:
    gr.Markdown("# 🤖 Batch Multi-Agent Exam Grading System")
    gr.Markdown("**Upload multiple student files and optional rubric, then click 'Grade All' - System grades all students**")
    
    with gr.Row():
        with gr.Column():
            # Multiple file upload
            gr.Markdown("**Upload multiple student responses, exam PDFs, or audio files. First file can contain exam questions.**")
            student_files = gr.File(
                label="📄 Upload Student Files (Multiple)",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md", ".mp3", ".wav", ".mp4"]
            )
            gr.Markdown("**Single rubric applied to all students. Can also contain exam questions if not in first file.**")
            rubric_file = gr.File(
                label="📋 Rubric (optional) - Can also contain exam questions",
                file_types=[".pdf", ".txt", ".md"]
            )
            
            submit_btn = gr.Button("🚀 Grade All", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        
        with gr.Column():
            summary_output = gr.Textbox(
                label="📊 Batch Summary",
                lines=3,
                max_lines=5,
                interactive=False
            )
            results_table = gr.HTML(
                label="📋 Results Table",
                value="<p>Upload files and click 'Grade All' to see results.</p>"
            )
            
            orchestration_metadata = gr.Textbox(
                label="🤖 Multi-Agent Orchestration Metadata",
                lines=20,
                max_lines=30,
                interactive=False,
                info="Shows triage results, agent routing, quality checks, and handoff information for all students"
            )
            
            gr.Markdown("### 📥 Downloads")
            
            with gr.Accordion("Individual Student Downloads", open=False):
                gr.Markdown("**JSON Files:**")
                individual_json_downloads = gr.File(
                    label="Download Individual JSON Files",
                    file_count="multiple"
                )
                gr.Markdown("**PDF Files:**")
                individual_pdf_downloads = gr.File(
                    label="Download Individual PDF Files",
                    file_count="multiple"
                )
            
            combined_download = gr.File(
                label="📦 Combined Results (JSON)",
                visible=True
            )
    
    def handle_submit(student_files, rubric_file):
        """Handle batch grading submission."""
        table_html, summary, metadata_text, individual_json, individual_pdf, combined = handle_batch_grading(
            student_files, rubric_file
        )
        
        # Convert dict to list for Gradio File component
        json_list = list(individual_json.values()) if individual_json else []
        pdf_list = list(individual_pdf.values()) if individual_pdf else []
        
        return (
            summary,
            table_html,
            metadata_text,
            json_list if json_list else None,
            pdf_list if pdf_list else None,
            combined
        )
    
    # Manual submit button
    submit_btn.click(
        fn=handle_submit,
        inputs=[student_files, rubric_file],
        outputs=[summary_output, results_table, orchestration_metadata, individual_json_downloads, individual_pdf_downloads, combined_download]
    )
    
    def clear_all():
        return None, None, "", "<p>Upload files and click 'Grade All' to see results.</p>", "", None, None, None
    
    clear_btn.click(
        fn=clear_all,
        outputs=[student_files, rubric_file, summary_output, results_table, orchestration_metadata, individual_json_downloads, individual_pdf_downloads, combined_download]
    )
    
    demo.launch(share=True)

