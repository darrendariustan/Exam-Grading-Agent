"""
Enhanced VC Pitch Agent with Function Calling
Converts VC pitch grader to a true agent with structured output
"""

import os
import json
import openai
import librosa
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


def is_mock_mode_enabled() -> bool:
    return os.getenv("MOCK_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

CACHE_DIR = os.path.join(os.getcwd(), "cache")
if not os.path.isdir(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

# Define function schema for structured output
def get_vc_pitch_functions():
    return [
        {
            "name": "grade_vc_pitch",
            "description": "Grade a VC pitch across four dimensions: Problem Clarity, Market Evidence, Solution Differentiation, and Delivery & Pacing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Problem": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Problem Clarity score (1-10)"
                    },
                    "Market": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Market Evidence score (1-10)"
                    },
                    "Solution": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Solution Differentiation score (1-10)"
                    },
                    "Delivery": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Delivery & Pacing score (1-10)"
                    },
                    "Feedback": {
                        "type": "string",
                        "description": "Actionable feedback for each dimension"
                    }
                },
                "required": ["Problem", "Market", "Solution", "Delivery", "Feedback"]
            }
        }
    ]

RUBRIC = """
You are a seasoned VC pitch grader. For a {duration}-minute audio pitch, give each dimension a score from 1 (poor) to 10 (excellent), using the following anchors:

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
   • 1–3: Monotone or too fast/slow (outside 80–200 WPM), frequent long pauses (>30%)  
   • 4–6: Understandable but some pacing issues (WPM 90–210, pauses 20–30%)  
   • 7–8: Good pace (110–160 WPM), pauses <20%  
   • 9–10: Engaging tone, ideal pacing (120–150 WPM), minimal pauses (<10%)
"""

def transcribe(mp3_path: str) -> str:
    """Return transcript from Whisper, caching results."""
    cache_file = os.path.join(CACHE_DIR, os.path.basename(mp3_path) + ".txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    with open(mp3_path, "rb") as audio_file:
        resp = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    transcript = resp if isinstance(resp, str) else resp.get("text", "")
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    return transcript

def audio_metrics(mp3_path: str):
    """Return words-per-minute, silence ratio, transcript, duration"""
    y, sr = librosa.load(mp3_path, sr=16000, mono=True)
    duration = len(y) / sr

    transcript = transcribe(mp3_path)
    words = transcript.split()
    wpm = len(words) / (duration / 60) if duration else 0

    segments = librosa.effects.split(y, top_db=30)
    voiced = sum((end - start) for start, end in segments) / sr
    silence_ratio = (duration - voiced) / duration if duration else 0

    return wpm, silence_ratio, transcript, duration

def grade_pitch_agent(mp3_path: str) -> Dict[str, Any]:
    """
    Grade VC pitch using function calling (true agent pattern).
    
    Args:
        mp3_path: Path to audio file
        
    Returns:
        Dict with Problem, Market, Solution, Delivery scores and Feedback
    """
    if is_mock_mode_enabled():
        return {
            "Problem": 7,
            "Market": 6,
            "Solution": 7,
            "Delivery": 8,
            "Feedback": "Mock feedback: sharpen market sizing evidence and add a clearer moat statement.",
            "mock_mode": True,
            "source_audio": os.path.basename(mp3_path)
        }

    wpm, silence, transcript, duration = audio_metrics(mp3_path)
    
    prompt = f"""
Pitch transcript:
{transcript}

Audio metrics:
• Words-per-minute: {wpm:.1f}
• Pause ratio: {silence:.1%}
• Duration: {duration/60:.1f} minutes

{RUBRIC.format(duration=duration/60)}
"""
    
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful VC pitch grader. Use the provided function to return structured scores."},
            {"role": "user", "content": prompt}
        ],
        tools=[{"type": "function", "function": f} for f in get_vc_pitch_functions()],
        tool_choice={"type": "function", "function": {"name": "grade_vc_pitch"}},
        temperature=0
    )
    
    # Extract function call result
    if resp.choices[0].message.tool_calls:
        function_args = resp.choices[0].message.tool_calls[0].function.arguments
        return json.loads(function_args)
    else:
        # Fallback to parsing message content
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON", "raw": raw}

# Backward compatibility
def grade_pitch(mp3_path: str) -> Dict[str, Any]:
    """Backward compatible wrapper."""
    return grade_pitch_agent(mp3_path)

