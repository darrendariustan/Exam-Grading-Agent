import openai
import json
import re

# 1. Set OpenAI API Key
import os
from dotenv import load_dotenv
load_dotenv()   # looks for a file named “.env” in cwd


def is_mock_mode_enabled() -> bool:
  return os.getenv("MOCK_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


if os.getenv("OPENAI_API_KEY"):
  openai.api_key = os.getenv("OPENAI_API_KEY")


def _estimate_question_count(text: str) -> int:
  numbered = re.findall(r"(?im)^\s*(?:q\s*\d+|question\s*\d+|\d+[\).:])", text or "")
  if numbered:
    return min(max(len(numbered), 1), 10)
  return 3


def _mock_grade_result(questions_markdown: str) -> dict:
  question_count = _estimate_question_count(questions_markdown)
  result = {}
  total = 0.0
  for i in range(1, question_count + 1):
    score = float(6 + (i % 3))
    total += score
    result[f"question_{i}"] = {
      "score": score,
      "feedback": "Mock feedback: clear response structure detected; add one concrete technical example."
    }
  result["total_score"] = round(total, 2)
  result["mock_mode"] = True
  return result

# 2. Grading function (LLM-powered)
def grade_exam(questions_markdown, answers_text, rubric_markdown=None, model="gpt-4-0125-preview"):
  if is_mock_mode_enabled():
    return _mock_grade_result(questions_markdown)

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

    # 4. Parse LLM response (strip markdown and "json" prefix)
    content = response.choices[0].message.content.strip()
    
    # Remove markdown code blocks
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) > 1:
            content = parts[1]
            # Remove language identifier if present (e.g., "json\n")
            if content.startswith("json"):
                content = content[4:].lstrip()
            content = content.strip()
    
    # Remove "json" prefix if present (some models add this)
    if content.lower().startswith("json"):
        content = content[4:].lstrip()
    
    # Try to find JSON object in content
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from content if it's embedded in text
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                print("Error: Could not parse JSON from LLM response.")
                print("Response:", content[:500])  # Print first 500 chars
                return {"error": "Could not parse JSON from LLM response", "raw": content[:500]}
        else:
            print("Error: Could not parse JSON from LLM response.")
            print("Response:", content[:500])  # Print first 500 chars
            return {"error": "Could not parse JSON from LLM response", "raw": content[:500]}

    return result
