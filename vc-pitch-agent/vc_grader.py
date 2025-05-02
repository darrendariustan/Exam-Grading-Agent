import os
import sys
import json
import openai
import librosa
import soundfile as sf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create cache directory
CACHE_DIR = os.path.join(os.getcwd(), "cache")
if not os.path.isdir(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

# Grading rubric
RUBRIC = """
You are a seasoned VC pitch grader. For a {{duration}}-minute audio pitch, give each dimension a score from 1 (poor) to 10 (excellent), using the following anchors:

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
{
  "Problem": <1–10>,
  "Market": <1–10>,
  "Solution": <1–10>,
  "Delivery": <1–10>,
  "Feedback": "<one sentence actionable feedback for each anchor>"
}
"""

# 1 – Transcribe with caching
def transcribe(mp3_path: str) -> str:
    """Return transcript from Whisper, caching results."""
    cache_file = os.path.join(CACHE_DIR, os.path.basename(mp3_path) + ".txt")
    if os.path.exists(cache_file):
        return open(cache_file).read()

    with open(mp3_path, "rb") as audio_file:
        resp = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    # resp is a string when response_format="text"
    transcript = resp if isinstance(resp, str) else resp.get("text", "")
    with open(cache_file, "w") as f:
        f.write(transcript)
    return transcript

# 2 – Compute audio metrics
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

# 3 – Build prompt
def build_prompt(transcript: str, wpm: float, silence: float, duration: float) -> str:
    return f"""
Pitch transcript:
{transcript}

Audio metrics:
• Words-per-minute: {wpm:.1f}
• Pause ratio: {silence:.1%}

{RUBRIC.replace("{{duration}}", f"{duration/60:.1f}")}
"""

# 4 – Grade the pitch
def grade_pitch(mp3_path: str) -> dict:
    wpm, silence, transcript, duration = audio_metrics(mp3_path)
    prompt = build_prompt(transcript, wpm, silence, duration)

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful pitch grader."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0
    )
    raw = resp.choices[0].message.content
    single = raw.replace("\n", " ")
    try:
        return json.loads(single)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw": raw}

# 5 – Entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python vc_grader_v3.py <path/to/pitch.mp3>")
        sys.exit(1)

    result = grade_pitch(sys.argv[1])
    print(json.dumps(result, indent=2))
