# Multi-Agent Lab

A configurable Streamlit platform for running structured, multi-agent annotation and review workflows with LLMs.

## Overview

Multi-Agent Lab lets you define:

- task setup and modality
- input and output schemas
- agent roster and interaction protocol
- instructions and per-agent prompt overrides
- dataset mappings

The run loop executes iterative agent turns (gossip-style), tracks history and label changes, and exports run outputs for analysis.

## Current Architecture

- app: Streamlit UI and experiment workflow pages
- core: runtime logic (agent, hub, protocol, state)
- experiments: output artifacts and saved runs
- literature: research references

## Repository Structure

```
ai-platform/
├── app/
│   ├── Main.py
│   ├── components/
│   └── pages/
│       ├── 1_Welcome.py
│       ├── 2_Agent Setup.py
│       ├── 3_Instructions.py
│       ├── 4_Dataset.py
│       ├── 5_Review.py
│       └── 6_Run.py
├── core/
│   ├── agent.py
│   ├── hub.py
│   ├── state.py
│   └── protocols/
│       └── gossip.py
├── experiments/
├── literature/
├── experiment_draft.json
├── instructions.txt
├── output_schema.json
├── output_schema_small.json
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- **Python 3.11** with packages listed in `requirements.txt`
- **OpenAI API access** (for GPT-4o)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Paularossi/ai-platform.git
   cd ai-platform
   ```

2. **Set up a virtual environment (venv)**

   Create the virtual environment with:
   ```bash
   python -m venv venv
   ```

   And then activate it with:
   ```bash
   venv\Scripts\activate.bat
   ```

3. **Install Python dependencies (change your CUDA version):**
   ```bash
   pip install -r requirements.txt
   ```

4. Run app

   ```bash
   streamlit run app/Main.py
   ```

## Dataset and Modality Behavior

### Text-only modality

- Dataset flow is text-only.
- Prompt user content contains mapped non-image fields.

### Image plus text modality

- ZIP upload can include images and spreadsheet metadata.
- Images are extracted to a temporary directory at load time.
- Column mapping determines which dataset column corresponds to each input field.
- Image matching uses normalized keys for robust resolution from metadata IDs to image files.

## Prompt Construction

At run time:

- all mapped non-image input fields are added to the user text block
- image content is attached separately as multimodal image input
- if no image can be resolved, run continues in text-only mode for that item

Practical implication:

- for fields like caption and brand, map columns and they appear in prompt text
- for image fields, map the image ID column so the file can be resolved and attached

## Output Parsing

The parser supports common structured variants, including:

- FIELD_NAME style blocks
- Q1 [field_name] style blocks
- verdict, probabilities, confidence keys
- pros and cons with or without bullet markers

## Troubleshooting

### Raw response parse failed

- model output can drift from expected structure
- parser accepts multiple formats, but edge cases can still occur
- inspect raw block shown in UI and refine instructions if needed


## Roadmap

- broader protocol set (crowd, duel, court)
- additional model providers
- richer evaluation and analytics dashboards
- improved experiment tracking and reproducibility metadata
- broad use cases (not just ad annotations)