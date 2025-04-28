# Experiments: Synthetic Data Generation & Evaluation

This folder contains scripts for running batch experiments and evaluations on your RAG pipeline using LangSmith.

## Contents
- `evaluate_on_dataset.py`: Runs your RAG pipeline on all questions in the LangSmith dataset and logs predictions.
- `evaluate_predictions.py`: Runs automated evaluation (Correctness, Helpfulness, Dopeness) on predictions using LangSmith evaluators.

## Prerequisites
- Python 3.10+
- All project dependencies installed (see project root requirements)
- API keys set as environment variables:
  - `OPENAI_API_KEY`
  - `LANGCHAIN_API_KEY`
- (Optional) **Vectorstore location:**
  - `VECTORSTORE_PATH` (default: `/tmp/vectorstore`)
- **LangSmith Tracing:**
  - `LANGCHAIN_TRACING_V2` (must be set to `true` to enable tracing in LangSmith)

## Usage

1. **Run the RAG pipeline and log predictions:**
   ```sh
   export OPENAI_API_KEY=sk-...
   export LANGCHAIN_API_KEY=ls-...
   export LANGCHAIN_TRACING_V2=true
   export VECTORSTORE_PATH=/tmp/vectorstore  # or your preferred path
   python evaluate_on_dataset.py
   ```
   This will process all questions in the LangSmith dataset and log your app's predictions.

2. **Run evaluation on predictions:**
   ```sh
   python evaluate_predictions.py
   ```
   This will score your predictions for correctness, helpfulness, and dopeness, and log results to LangSmith.

3. **View Results:**
   - Go to your [LangSmith dashboard](https://smith.langchain.com/) and open the relevant project/dataset to see experiment results and metrics.

## Notes
- Make sure your dataset name matches between scripts and LangSmith.
- You can rerun these scripts as you update your pipeline or data.
- The vectorstore will be stored in `/tmp/vectorstore` by default, which is suitable for cloud environments like Hugging Face Spaces. Set `VECTORSTORE_PATH` if you want to use a different location.
- **Tracing:** Setting `LANGCHAIN_TRACING_V2=true` is required for detailed trace logging in LangSmith. Without this, traces will not appear in your LangSmith dashboard. 