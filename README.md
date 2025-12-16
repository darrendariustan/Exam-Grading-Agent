# Agentic AI for Exam Grading (Master Project)

This repository contains a unified multi‑agent exam grading system with three specialist agents (technical, narrative, and VC‑pitch) orchestrated through automatic triage, guardrails, and handoff mechanisms. Each subfolder contains standalone examples; the `multi‑agent` folder provides the full orchestrated system.

> Agents 4 Education showcases how LLM‑powered agents can collaborate to automate formative assessment across different exam types, as part of a Master’s in Business Analytics capstone project.

## 🎥 Demo

Watch the demo video to see the multi-agent grading system in action:

[Demo Video](Demo_Multi-Grading_Agent.mp4)

Or view directly: [Click Here](https://www.youtube.com/watch?v=fka04b7ig90)

## Technical Implementation Breakdown

This repository is structured into specialized specialist agents and a central orchestration layer. Below is the technical breakdown of each component:

### 📂 `multi-agent/` (Orchestration Core)
The brain of the system, handling lifecycle management from input to final graded report.
- **`agent_orchestrator.py`**:
  - **Triage Mechanism**: Uses `gpt-4o-mini` with zero-shot prompting to classify inputs into *Technical*, *Narrative*, or *VC Pitch* based on content analysis (e.g., detecting equations vs. essays).
  - **Guardrails**: Implements pre-processing checks for content length, format validity, and basic safety (e.g., script injection detection).
  - **Handoff Logic**: Implements a resilience pattern where `Technical` failures automatically fallback to `Narrative` grading to ensure a response is always returned.
- **`batch_grading_ui.py`**:
  - Built with **Gradio**, supporting concurrent multi-file uploads.
  - Features intelligent file splitting (separating Questions from Answers automatically).
  - Generates individual PDF reports using `fpdf` and aggregates batch statistics.

### 📂 `technical-agent/` (STEM Specialist)
Focused on precision and reasoning for Math, Science, and Coding exams.
- **`tech_grading_agent.py`**:
  - **Model**: Leverages `gpt-4-0125-preview` (GPT-4 Turbo) for superior reasoning capabilities required for checking calculations and logic.
  - **Prompting**: Uses strict JSON-mode prompting to ensure output consistency for downstream parsing.
- **`pdf_to_markdown.py`**:
  - **Extraction**: Uses `pdfplumber` to extract text and **preserve table structures**, converting them to Markdown tables so the LLM can "see" data grids correctly.

### 📂 `narrative-agent/` (Humanities Specialist)
Optimized for essays, case studies, and subjective feedback.
- **`exam_grader_agents.py`**:
  - **Structured Output**: Uses **OpenAI Function Calling** (`generate_exam_responses`) instead of raw JSON prompts to guarantee schema adherence for scoring.
  - **Metrics Tracking**: Maintains a local `grading_metrics_history.pkl` to track model performance (MAE, RMSE) over time.
  - **Reporting**: Generates professional-grade PDFs using `reportlab`.

### 📂 `vc-pitch-agent/` (Audio/Multimodal Specialist)
A pipeline for evaluating spoken presentations.
- **`vc_grader_agent.py`**:
  - **Audio Processing**: Uses `librosa` to compute acoustic features:
    - **WPM (Words Per Minute)**: To measure pacing.
    - **Silence Ratio**: To detect hesitation or confidence.
  - **Transcription**: Calls OpenAI's **Whisper** model for high-fidelity speech-to-text.
  - **Grading**: Uses `gpt-4o-mini` with a specialized rubric for "Problem," "Market," "Solution," and "Delivery".

### Root Files
- **`.env`**: Stores `OPENAI_API_KEY` and other sensitive configurations.
- **`requirements.txt`**: Lists core dependencies: `openai`, `gradio`, `pdfplumber`, `librosa`, `reportlab`, `fpdf`.

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


First, change to the directory of the desired prototype:

```bash
cd multi-agent  # For the web interface
```

Run the example with this command:

```bash
# Batch grading interface (multiple students at once):
python batch_grading_ui.py
```

> **Note:** `batch_grading_ui.py` is the main and only interface available. Other UI files in this directory are legacy versions and should not be used.

#### Interface Features:

- **`batch_grading_ui.py`**  
  Batch grading interface that allows uploading multiple student files at once. Features:
  - Upload multiple student responses, exam PDFs, or audio files
  - Single rubric applied to all students
  - Results displayed in a table format
  - Individual and combined download options (JSON/PDF)
  - Multi-agent orchestration metadata showing triage, routing, and handoff information

## How the Multi-Agent System Works

The system uses a true multi-agent architecture with orchestration, not just a single LLM with different prompts. Here's how it works:

### 1. **Triage Agent** 🤖
- **Purpose**: Automatically classifies the exam type before grading
- **Process**: Analyzes the input content (exam questions and student responses) to determine if it's:
  - **Technical**: Factual knowledge, mathematical reasoning, problem-solving
  - **Narrative**: Open-ended analysis, strategic thinking, reflective writing
  - **VC Pitch**: Audio-based entrepreneurial/product presentations
- **Output**: Exam type classification with confidence score and reasoning

### 2. **Guardrails** ✅
- **Purpose**: Input validation and quality checks
- **Process**: 
  - Validates that exam questions and student responses are present
  - Checks content length and quality
  - Detects input type (text vs. audio)
  - Issues warnings for potential issues (e.g., very short responses)
- **Output**: Quality assessment with warnings if needed

### 3. **Agent Router** 🎯
- **Purpose**: Routes the exam to the appropriate specialist agent
- **Process**: Based on triage results, routes to:
  - **Technical Agent** (`technical-agent/tech_grading_agent.py`): For technical exams
  - **Narrative Agent** (`narrative-agent/exam_grader_agents.py`): For narrative/essay exams
  - **VC Pitch Agent** (`vc-pitch-agent/vc_grader_agent.py`): For audio pitch grading
- **Output**: Grading results from the specialist agent

### 4. **Handoff Mechanism** 🔄
- **Purpose**: Error recovery and fallback when an agent fails
- **Process**: 
  - If the primary agent fails or returns an error, the system automatically tries a fallback agent
  - For example: If technical agent fails → tries narrative agent as fallback
  - Preserves error information for debugging
- **Output**: Grading results from fallback agent or error details

### 5. **Orchestration Metadata** 📊
- **Purpose**: Provides transparency into the multi-agent decision-making process
- **Contains**:
  - Triage results (exam type, confidence, reasoning)
  - Agent used (technical/narrative/VC pitch)
  - Quality check results
  - Handoff information (if applicable)
  - Status (success/error)

### Architecture Flow:

```
Input Files → Triage Agent → Guardrails → Agent Router → Specialist Agent → Results
                                      ↓
                              (if error) → Handoff → Fallback Agent → Results
```

This architecture ensures:
- **Automatic classification** without manual exam type selection
- **Specialized grading** by domain-specific agents
- **Error resilience** through handoff mechanisms
- **Transparency** through orchestration metadata
- **Quality assurance** through guardrails
