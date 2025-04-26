# SDG via LangGraph

This project reproduces the RAGAS Synthetic Data Generation steps using LangGraph instead of the Knowledge Graph approach.

## Features

- Synthetic data generation using Evol Instruct method
- Three evolution types: Simple, Multi-Context, and Reasoning
- Output includes evolved questions, answers, and relevant contexts
- Deployed as a Streamlit app on Hugging Face Spaces

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
- `ENVIRONMENT`: Set to "production" for production mode

## Project Structure

- `app.py`: Streamlit application for the Hugging Face deployment
- `preprocess/`: Code for preprocessing HTML files and creating embeddings
- `graph/`: LangGraph implementation for synthetic data generation
- `data/`: HTML files containing LLM evolution data
- `tests/`: Test files
- `generated/`: Generated documents and vectorstore
