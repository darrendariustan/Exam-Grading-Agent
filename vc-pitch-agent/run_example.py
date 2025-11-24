import os
import sys
import json
from vc_grader import grade_pitch

# Example: Grade a VC pitch audio file
# Make sure you have an audio file in the audio/ directory or provide a path to your file

# Default example file path
audio_file = "audio/1_Qualud.mp3"

# Check if file exists, if not, use the first available file in audio/ directory
if not os.path.exists(audio_file):
    audio_dir = "audio"
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
        if audio_files:
            audio_file = os.path.join(audio_dir, audio_files[0])
            print(f"Using audio file: {audio_file}")
        else:
            print("No .mp3 files found in audio/ directory")
            print("Usage: Place an .mp3 file in the audio/ directory or modify audio_file path")
            sys.exit(1)
    else:
        print(f"Audio file not found: {audio_file}")
        print("Usage: python run_example.py [path/to/pitch.mp3]")
        sys.exit(1)

# Allow command line argument for custom file
if len(sys.argv) > 1:
    audio_file = sys.argv[1]

print(f"Grading VC pitch: {audio_file}")
print("-" * 50)

# Grade the pitch
result = grade_pitch(audio_file)

# Output results
if result:
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print("Error: No result returned")
    sys.exit(1)

