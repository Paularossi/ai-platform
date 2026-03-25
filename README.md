# 🧠 Multi-Agent Lab 
A configurable platform for studying multi-agent AI interaction, consensus formation, and bias in structured tasks.

## Overview

This project develops a platform for studying multi-agent interaction among large language models (LLMs) in structured task environments. The central objective is to investigate whether iterative communication between multiple AI agents operating under predefined interaction protocols can improve task performance, consistency, and alignment compared to zero-shot model outputs.

Rather than treating model predictions as static, the platform models annotation and decision-making as a dynamic, multi-step process, where agents sequentially or collectively review and revise a shared structured state. This enables systematic analysis of convergence behavior, disagreement patterns, and sources of bias across agents and model providers.

While the initial application focuses on multimodal advertisement annotation (image and text), the system is designed to be task-agnostic, supporting a wide range of classification, evaluation, and revision tasks through configurable schemas and instruction sets.


## 📁 Repository Structure

```
ai-platform/
├── app/                                     # Streamlit UI for configuring and running experiments
│   ├── Main.py                              # App entry point
│   ├── components/                          # Placeholder for shared UI components (currently empty)
│   └── pages/                               # Individual Streamlit page definitions
│       ├── 1_Welcome.py
│       ├── 2_Agent Setup.py
│       ├── 3_Instructions.py
|       └── ...                              # add more pages
│
├── core/                                    # Planned orchestration + agent logic (empty)
├── experiments/                             # Saved experiment configurations and results (empty)
├── literature/                              # Reference papers on multi-agent systems and consensus (update this)
│   └── ...
│
├── experiment_draft.json                    # Example experiment template saved from the app
├── instructions.txt                         # Prompt template for advertisement annotation tasks
├── output_schema.json                       # Sample schema for the output
├── requirements.txt                         # Python dependencies for the platform
└── README.md                                # this file
```


## 🚀 Getting Started

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

4. **Run the platform**
   ```bash
   streamlit run app/main.py
   ```
   A web page should open automatically.



## 📊 System Design

The platform is built around a modular architecture with clear separation of concerns:

- **Frontend (Streamlit)** - A user interface for defining experiments, configuring tasks, and visualizing agent interactions.

- **Orchestration Layer** - Protocol-driven execution of multi-agent workflows, including sequential (gossip), parallel (crowd), and structured interaction settings (duel, court).

- **Agent Layer** - Role-based agents (e.g., initializer, skeptical reviewer, conservative validator) that iteratively modify or validate a shared output.

- **Provider Layer** - Pluggable LLM backends (initially OpenAI, with planned support for additional providers).

- **State Management** - A shared, structured state object that evolves over time, accompanied by a full revision history for each interaction step.

- **Logging and Reproducibility** - Configuration-driven experiments with stored metadata, intermediate outputs, and complete execution traces to enable reproducible analysis.


## 🗺️ Roadmap

- ✅ Initial project structure
- ✅ Streamlit app setup 
- ✅ Experiment setup UI (task + schema + instructions)
- ✅ High-level architecture defined
- ✅ Agent and protocol configuration interface
- ⬜ Formal definition of experiment configuration and shared state schemas
- ⬜ Implementation of core orchestration logic (sequential gossip protocol)
- ⬜ Additional interaction protocols (crowd, duel, court)
- ⬜ Multi-provider integration (e.g., Anthropic, Gemini, local models)
- ⬜ Evaluation metrics (consensus, flexibility, bias measures)
- ⬜ Analysis and visualization dashboard
- ⬜ Optional backend and database support for scalability


## Objective

The long-term goal is to develop a research-oriented framework for multi-agent AI systems, enabling systematic experimentation with interaction protocols, reproducible evaluation of model behavior, and deeper understanding of how collective AI processes influence decision quality and bias.