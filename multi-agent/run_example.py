# test.py

import os
import asyncio

# 1) Load .env into os.environ
from dotenv import load_dotenv
load_dotenv()   # looks for a file named “.env” in cwd

# 2) confirm we have the key
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in environment"

# 3) Now import and run your agents
from exam_grader_agents import grade_exam, evaluation_to_csv

async def main():
    # read the student answers markdown into a string
    with open("Respostes_MD.md", encoding="utf-8") as f:
        md_blob = f.read()

    input_data = {
        "pdf_path": "CRA_Final_Examen_Gener_2025_CATALÀ.pdf",
        "rubric_path": "CRA_Final_Examen_Rubric.pdf",
        # pass the markdown text directly
        "md_text": md_blob,
    }

    evaluation = await grade_exam(input_data)

    # print(evaluation.to_input_list())
    # print(evaluation.raw_responses)

    # print JSON
    import json
    print(json.dumps(evaluation.dict(), indent=2, ensure_ascii=False))

    # write CSV
    out_csv = "student123_exam1_grades.csv"
    evaluation_to_csv(evaluation, out_csv)
    print("Wrote grades to", out_csv)

if __name__ == "__main__":
    asyncio.run(main())
