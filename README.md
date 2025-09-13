# Agentic AI for Exam Grading (Master Project)

This repository collects three prototype exam‑grading systems—one each for technical, narrative/strategy and VC‑pitch exams—and a work‑in‑progress multi‑agent orchestration built with the OpenAI Agents SDK. Each subfolder contains an independent end‑to‑end example; the `multi‑agent` folder shows how they can be combined with triage, guardrails and handoffs.

> Agents 4 Education showcases how LLM‑powered agents can collaborate to automate formative assessment across different exam types, as part of a Master’s in Business Analytics capstone project.

## Repository layout

- **multi‑agent/**  
  A WIP unified system that classifies an exam, applies preprocessing tools (PDF→Markdown, audio analysis), enforces guardrails and then delegates to the appropriate specialist agent.  
- **technical‑agent/**  
  A standalone grader for factual knowledge, mathematical reasoning, and technical problem-solving skills.​
- **narrative‑agent/**  
  Tailored for exams involving open-ended responses, strategic thinking, or reflective writing—particularly suited to humanities or business-oriented subjects.
- **vc‑pitch‑agent/**
  Focused on grading non-written final projects, such as live or recorded presentations, especially those centered around entrepreneurial or product development ideas.​
- **.env**, **requirements.txt**, **README.md**  
  Root‑level environment setup, dependencies and this guide.

## Getting started

You can replicate on your machine using the following steps.


### 1. Clone the repository

```bash
git clone https://github.com/darrendariustan/Exam-Grading-Agent.git
cd Agents_4_Education
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

You’ll need a `.env` file in the root of the project with the Open AI API Key:

```
# Open AI API Key
OPENAI_API_KEY=your_openai_api_key
```

> ⚠️ Important
>
> Be aware of API costs!



### 5. Running the examples

Each folder has its own `run_example.py`.

First, change to the directory of the desired prototype:

```bash
cd multi-agent  # For the web interface
# OR
cd narrative-agent  # For the narrative agent example
# OR
cd technical-agent  # For the technical agent example
```

Run the example with this command:

```bash
# For multi-agent web interface:
python gradio_ui_1.py

# For other agents:
python run_example.py