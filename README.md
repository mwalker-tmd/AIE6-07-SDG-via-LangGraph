# SDG via LangGraph

This project reproduces the RAGAS Synthetic Data Generation steps using LangGraph instead of the Knowledge Graph approach.

## Features

- Synthetic data generation using Evol Instruct method
- Three evolution types: Simple, Multi-Context, and Reasoning
- Output includes evolved questions, answers, and relevant contexts
- Deployed as a Streamlit app on Hugging Face Spaces

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the app at `http://localhost:8501`

## Deployment to Hugging Face Spaces

1. Create a new Space on Hugging Face with Streamlit as the SDK
2. Upload all files to the Space
3. Set the following environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ENVIRONMENT`: Set to "production" for production mode

## Project Structure

- `app.py`: Streamlit application for the Hugging Face deployment
- `preprocess/`: Code for preprocessing HTML files and creating embeddings
- `graph/`: LangGraph implementation for synthetic data generation
- `data/`: HTML files containing LLM evolution data
