# VC Pitch Agent

A standalone grader for VC pitch presentations in audio format. This agent evaluates audio pitches based on four key dimensions: Problem Clarity, Market Evidence, Solution Differentiation, and Delivery & Pacing.

## Features

- **Audio Transcription**: Uses OpenAI Whisper API to transcribe audio pitches
- **Audio Analysis**: Computes words-per-minute (WPM) and silence ratio using librosa
- **Intelligent Grading**: Uses GPT-4o-mini to grade pitches on four dimensions (1-10 scale)
- **Caching**: Caches transcriptions to avoid redundant API calls

## Dependencies

All dependencies are listed in the root `requirements.txt` file:
- `openai` - For Whisper transcription and GPT grading
- `librosa` - For audio analysis (WPM, silence detection)
- `python-dotenv` - For environment variable management

## Usage

### Basic Usage

```bash
python vc_grader.py <path/to/pitch.mp3>
```

### Example

```bash
python run_example.py
# Or with a specific file:
python run_example.py audio/1_Qualud.mp3
```

## Output Format

The grader returns a JSON object with the following structure:

```json
{
  "Problem": 8,
  "Market": 7,
  "Solution": 9,
  "Delivery": 6,
  "Feedback": "One sentence actionable feedback for each dimension"
}
```

## Grading Criteria

1. **Problem Clarity** (1-10): How clear and compelling the problem statement is
2. **Market Evidence** (1-10): Quality and quantity of market data cited
3. **Solution Differentiation** (1-10): How well the solution is differentiated from competitors
4. **Delivery & Pacing** (1-10): Quality of presentation delivery, including WPM and pause management

## Cache

Transcripts are cached in the `cache/` directory to speed up repeated grading of the same files.
