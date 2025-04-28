# exam_grader_agents.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import pdfplumber
import librosa

from agents import (
    Agent,
    Runner,
    handoff,
    function_tool,
    InputGuardrail,
    GuardrailFunctionOutput,
    input_guardrail,
)

# import the wrapper for non-strict schemas:
from agents.agent_output import AgentOutputSchema

#
# 1. Preprocessing tools
#

@function_tool
def extract_pdf_to_markdown(pdf_path: str) -> str:
    """
    Extracts text and tables from a PDF and returns markdown.
    """
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

    def convert_table_to_markdown(table: List[List[str]]) -> str:
        header, *rows = table
        md = "| " + " | ".join(header) + " |\n"
        md += "| " + " | ".join("—" for _ in header) + " |\n"
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

def transcribe(mp3_path: str) -> str:
    with open(mp3_path, "rb") as f:
        resp = openai.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="text"
        )
    return resp

@function_tool
def analyze_audio(mp3_path: str) -> Dict[str, Any]:
    """
    Computes duration, WPM, silence ratio, and returns transcript.
    """
    y, sr = librosa.load(mp3_path, sr=16000, mono=True)
    duration = len(y) / sr
    # transcribe() would call openai.audio.transcriptions.create(...)
    # here we assume transcribe() is available in scope
    transcript = transcribe(mp3_path)
    word_count = len(transcript.split())
    wpm = word_count / (duration / 60)
    intervals = librosa.effects.split(y, top_db=30)
    voiced = sum((e - s) for s, e in intervals) / sr
    silence_ratio = (duration - voiced) / duration
    return {"duration": duration, "wpm": wpm, "silence_ratio": silence_ratio, "transcript": transcript}


# 2. Pydantic schemas for structured outputs
#

class PartEvaluation(BaseModel):
    part_name: str
    score: float
    max_score: float
    feedback: str

class ExamEvaluation(BaseModel):
    exam_type: str
    overall_score: float
    parts: List[PartEvaluation]
    audio_metrics: Optional[Dict[str, Any]] = None

#
# 3. Guardrail to require audio for VC pitches
#

class VCAudioCheck(BaseModel):
    has_audio: bool
    reasoning: str

guardrail_agent = Agent(
    name="VC Pitch Audio Guardrail",
    instructions="Check whether the provided inputs include an audio file for a VC‐pitch exam.",
    # wrap in AgentOutputSchema to disable strict mode
    output_type=AgentOutputSchema(VCAudioCheck, strict_json_schema=False)
)  # :contentReference[oaicite:0]{index=0}

# @input_guardrail()
async def require_audio(ctx, agent, inputs):
    # Run the guardrail agent on the same inputs
    result = await Runner.run(guardrail_agent, inputs, context=ctx.context)
    out = result.final_output_as(VCAudioCheck)
    return GuardrailFunctionOutput(
        output_info=out.has_audio,
        tripwire_triggered=not out.has_audio
    )

#
# 4. Specialist grading agents
#

technical_agent = Agent(
    name="Technical Examiner",
    handoff_description="Specialist in grading technical problem‐solving exams",
    instructions=(
        "You are an expert grader for technical exams. "
        "Given the question, the student’s answer, and an optional rubric (in markdown), "
        "assign scores for correctness, depth, and clarity, then justify each score."
    ),
    # wrap ExamEvaluation to disable strict JSON schema
    output_type=AgentOutputSchema(ExamEvaluation, strict_json_schema=False),
    tools=[extract_pdf_to_markdown],  # for PDF rubrics or questions :contentReference[oaicite:1]{index=1}
)

narrative_agent = Agent(
    name="Narrative Examiner",
    handoff_description="Specialist in grading narrative and strategy essays",
    instructions=(
        "You are an expert grader for narrative/strategy essays. "
        "Evaluate structure, argumentation, creativity, and evidence use. "
        "Score each dimension and provide feedback."
    ),
    output_type=AgentOutputSchema(ExamEvaluation, strict_json_schema=False),
    tools=[extract_pdf_to_markdown],
)

vc_pitch_agent = Agent(
    name="VC Pitch Examiner",
    handoff_description="Specialist in grading VC-pitch presentations",
    instructions=(
        "You are an expert grader for VC pitch presentations. "
        "Using the transcript, delivery metrics, and any slides, score problem definition, solution, market, team, "
        "and presentation style. Provide detailed feedback."
    ),
    output_type=AgentOutputSchema(ExamEvaluation, strict_json_schema=False),
    tools=[analyze_audio],
    # apply guardrail here: if the student hasn't provided mp3, tripwire
    input_guardrails=[ InputGuardrail(guardrail_function=require_audio) ],
)

#
# 5. Triage agent with handoffs and guardrail
#

# triage_agent = Agent(
#     name="Exam Triage Agent",
#     instructions=(
#         "Determine the exam_type ∈ {technical, narrative, vc_pitch} from the inputs. "
#         "Then hand off to the corresponding specialist grading agent."
#     ),
#     handoffs=[technical_agent, narrative_agent, vc_pitch_agent],  # :contentReference[oaicite:2]{index=2}
# )

from agents import Agent, handoff
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

triage_agent = Agent(
    name="Exam Triage Agent",
    # this helper injects a prefix that tells the model how to use handoffs ◆
    instructions=prompt_with_handoff_instructions(
        """You have three specialist agents:
         - transfer_to_Technical_Examiner  
         - transfer_to_Narrative_Examiner  
         - transfer_to_VC_Pitch_Examiner  

        Read the inputs and **invoke exactly one** of these tools,
        by emitting the single tool call whose name matches
        the specialist best suited. Do not call more than one."""
    ),
    handoffs=[
      handoff(technical_agent),
      handoff(narrative_agent),
      handoff(vc_pitch_agent),
    ],
)


#
# 6. Orchestration example
#

# exam_grader_agents.py  (only the grade_exam part shown)  
from agents.items import TResponseInputItem   # alias for ResponseInputItemParam :contentReference[oaicite:0]{index=0}
from agents import Runner
import json

async def grade_exam(input_data: dict) -> ExamEvaluation:
    """
    Build a list of TResponseInputItemParam and call Runner.run with it.
    input_data keys: pdf_path, rubric_path, md_path, optional mp3_path.
    """
    # 1) Start with an optional system message
    items: list[TResponseInputItem] = [
        {"role": "system", "content": "You will be given an exam, a rubric, and student responses. Decide exam_type then grade."}
    ]

    # 2) Add each file‐path or markdown blob as a separate user message
    for key in ("pdf_path", "rubric_path", "md_path"):
        if key in input_data:
            items.append({
                "role": "user",
                "content": f"{key}: {input_data[key]}"
            })

    # 3) Attach audio path if present
    if "mp3_path" in input_data:
        items.append({
            "role": "user",
            "content": f"mp3_path: {input_data['mp3_path']}"
        })

    # 4) Run the triage → specialist pipeline
    run = await Runner.run(triage_agent, items)

    # 5) Validate and return
    return run.final_output_as(ExamEvaluation)

#
# 7. Export helper
#

import pandas as pd

def evaluation_to_csv(ev: ExamEvaluation, path: str):
    rows = [
        {
            "Part": p.part_name,
            "Score": p.score,
            "Max Score": p.max_score,
            "Feedback": p.feedback
        }
        for p in ev.parts
    ]
    if ev.audio_metrics:
        rows.append({
            "Part": "Audio Metrics",
            "Score": "",
            "Max Score": "",
            "Feedback": str(ev.audio_metrics)
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
