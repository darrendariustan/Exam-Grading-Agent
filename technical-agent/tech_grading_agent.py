import openai
import json

# 1. Set OpenAI API Key
import os
from dotenv import load_dotenv
load_dotenv()   # looks for a file named “.env” in cwd
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in environment"

# 2. Grading function (LLM-powered)
def grade_exam(questions_markdown, answers_text, rubric_markdown=None, model="gpt-4-0125-preview"):
    # Prepare the prompt
    prompt = f"""
You are a grading assistant for technical exams. Your role is to evaluate student responses based on the provided questions and, if available, the rubric.

### Grading Instructions:
- For each question:
  - Read the question and the student's answer.
  - If a rubric is provided for that question, follow it carefully to assign points based on the expected criteria.
  - If no rubric is provided, use your expert-level knowledge of the subject to assess:
    - Factual accuracy.
    - Completeness.
    - Clarity.
- Assign a score for each answer:
  - Use the point scale from the rubric, or if none is provided, use a default scale of 0-10 points.
- Provide feedback for each answer:
  - Explain why the student received the score.
  - Offer suggestions for improvement if the answer is incomplete or incorrect.

### Output Format:
Respond ONLY with raw JSON (no markdown):
{{
  "question_1": {{
    "score": X,
    "feedback": "..."
  }},
  "question_2": {{
    "score": Y,
    "feedback": "..."
  }},
  ...
  "total_score": Z
}}

### Rubric (if available):
{rubric_markdown or "No rubric provided."}

### Exam Questions:
{questions_markdown}

### Student Answers:
{answers_text}
"""

    # 3. Call OpenAI LLM
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0,
        # seed=42
    )

    # 4. Parse LLM response (strip markdown)
    content = response.choices[0].message.content
    if content.startswith("```"):
        content = content.strip().split("```")[1]
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        print("Error: Could not parse JSON from LLM response.")
        print("Response:", content)
        return None

    return result
