from tech_grading_agent import grade_exam
from pdf_to_markdown import extract_pdf_to_markdown

questions_markdown = extract_pdf_to_markdown("technical-agent/CRA_Final_Examen_Gener_2025_CATALÀ.pdf")
rubric_markdown = extract_pdf_to_markdown("technical-agent/CRA_Final_Examen_Rubric.pdf")

with open("technical-agent/Respostes_MD.md", 'r', encoding='utf-8') as file:
    content = file.read()

# Student answers (for testing)
answers_text = content

# Grade the exam
grading_result = grade_exam(questions_markdown, answers_text, rubric_markdown)

# Output results
if grading_result:
    print(json.dumps(grading_result, indent=2))