import os
from dotenv import load_dotenv
from langsmith import Client
from graph.types import SDGState
from graph.build_graph import build_sdg_graph
from preprocess.embed_documents import create_or_load_vectorstore
from preprocess.html_to_documents import extract_documents_from_html
from langchain_openai import ChatOpenAI
from pathlib import Path
import pickle


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- CONFIG ---
DATASET_NAME = "State of AI Across the Years!"
PROJECT_NAME = "State of AI Across the Years!"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")

# --- SETUP ENV ---
os.environ["LANGCHAIN_PROJECT"] = PROJECT_NAME
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- LOAD DOCUMENTS & VECTORSTORE ---
def load_docs():
    output_file = Path("generated/documents.pkl")
    if output_file.exists():
        with open(output_file, "rb") as f:
            return pickle.load(f)
    # Fallback: extract from HTML
    docs = []
    data_dir = Path("data")
    for html_file in data_dir.glob("*.html"):
        docs.extend(extract_documents_from_html(str(html_file), label=html_file.stem))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(docs, f)
    return docs

def main():
    # Load dataset from LangSmith
    client = Client()
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    examples = client.list_examples(dataset_id=dataset["id"])

    # Load docs/vectorstore
    docs = load_docs()
    vectorstore_path = os.environ.get("VECTORSTORE_PATH", "/tmp/vectorstore")
    vectorstore = create_or_load_vectorstore(docs, path=vectorstore_path)
    llm = ChatOpenAI()
    graph = build_sdg_graph(docs, vectorstore, llm)

    # For each example, run the graph and log prediction
    for example in examples:
        question = example.inputs["question"]
        reference = example.outputs["answer"]
        # Prepare initial state
        state = SDGState(input=question)
        result = graph.invoke(state)
        if not isinstance(result, SDGState):
            result = SDGState(**dict(result))
        # Log prediction to LangSmith
        client.create_run(
            name="SDG App Run",
            inputs={"question": question},
            outputs={"output": result.answer},
            reference_outputs={"answer": reference},
            example_id=example.id,
            project_name=PROJECT_NAME,
        )
        print(f"Processed: {question}\n  → {result.answer}\n")

if __name__ == "__main__":
    main() 