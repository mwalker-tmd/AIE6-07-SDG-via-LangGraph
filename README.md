---
title: TMD-SDG-via-LangGraph
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# SDG via LangGraph

This project reproduces the RAGAS Synthetic Data Generation steps using LangGraph instead of the Knowledge Graph approach.

## Features

- Synthetic data generation using Evol Instruct methodology
- Iterative question evolution with alternating prompts:
  - Even iterations: More challenging and insightful questions
  - Odd iterations: More creative and original questions
- Consistent state management across iterations
- Standardized JSON output format with linked questions, answers, and contexts
- Deployed as a Streamlit app on Hugging Face Spaces

## Evol Instruct Implementation

This project implements the Evol Instruct methodology for evolving questions through multiple iterations. The implementation has several key aspects that should be considered when modifying the code:

### Core Principles

1. **Single Evolution Per Pass**: Each graph invocation performs one evolution step, maintaining clarity and control over the evolution process.
2. **Alternating Prompts**: The system alternates between:
   - Challenging/insightful prompts (even-numbered iterations)
   - Creative/original prompts (odd-numbered iterations)
3. **State Management**: Evolution history is preserved between iterations of the evolving questions process. In addition, each node in the chain only processes the latest evolved question.
4. **Configurable Evolution Count**: The number of evolution passes can be controlled through UI or environment variables, allowing flexibility in the evolution process.

### Implementation Details

- The evolution logic is implemented in `graph/nodes/evolve.py`
- Prompt selection is based on the number of existing evolutions
- State management ensures each evolution builds upon previous results
- Results maintain consistent IDs (`q0`, `q1`, etc.) across questions, answers, and contexts

### Configuration

- Number of evolution passes can be controlled via:
  - Streamlit UI slider (web interface)
  - `NUM_EVOLVE_PASSES` environment variable (CLI)

### ⚠️ Important Considerations

When modifying this codebase, please keep in mind:
1. The evolution process is intentionally sequential and builds upon previous iterations
2. Maintaining the alternating prompt pattern is crucial for question diversity
3. State management between iterations must preserve the evolution history
4. The ID system (`q0`, `q1`, etc.) must remain consistent across all collections

## Quick Start

### Local Development

1. Create a virtual environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

3. Run the application:
```bash
streamlit run app.py
```

4. Access the app at `http://localhost:8501`

## Deployment

### HuggingFace Spaces

1. Create a new Space on HuggingFace:
   - Go to https://huggingface.co/spaces
   - Click "New Space"
   - Choose "Streamlit" as the SDK
   - Choose "Docker" as the hardware

2. Add the HuggingFace remote:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

3. Push to HuggingFace:
```bash
git push hf main
```

### Environment Variables

The following environment variables need to be set in your HuggingFace Space settings:

- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGCHAIN_API_KEY`: Your LangChain API key (optional)
- `LANGCHAIN_PROJECT`: Your LangChain project name (optional)
- `LANGCHAIN_TRACING_V2`: Set to "true" to enable tracing
- `ENVIRONMENT`: Set to "production" for production mode
- `NUM_EVOLVE_PASSES`: Number of evolution iterations (default: 2)
- `VECTORSTORE_PATH`: Path to store vectors (default: /tmp/vectorstore)

## Project Structure

- `app.py`: Streamlit application for the Hugging Face deployment
- `main.py`: CLI interface with the same functionality as the web app
- `preprocess/`: Code for preprocessing HTML files and creating embeddings
- `graph/`: LangGraph implementation for synthetic data generation
  - `nodes/`: Individual graph nodes (evolve, retrieve, answer)
  - `types.py`: State management and data structures
  - `build_graph.py`: Graph construction and configuration
- `data/`: HTML files containing LLM evolution data
- `tests/`: Test files ensuring correct implementation
- `generated/`: Generated documents, vectorstore, and results
