import os
import pickle
import json
from preprocess.html_to_documents import extract_documents_from_html
from langchain.schema import Document
from pathlib import Path
from graph.types import SDGState
from preprocess.embed_documents import create_or_load_vectorstore
from graph.build_graph import build_sdg_graph
from langchain_openai import ChatOpenAI


class DocumentEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Document):
            return {
                "page_content": obj.page_content,
                "metadata": obj.metadata
            }
        if isinstance(obj, SDGState):
            return {
                "input": obj.input,
                "evolved_question": obj.evolved_question,
                "context": obj.context,
                "answer": obj.answer
            }
        return super().default(obj)


def is_dev_mode() -> bool:
    return os.getenv("ENVIRONMENT", "development").lower() == "development"


def get_data_paths():
    return [
        ("data/2023_llms.html", "llm-2023"),
        ("data/2024_llms.html", "llm-2024"),
    ]


def load_or_generate_documents() -> list[Document]:
    output_file = Path("generated/documents.pkl")
    if output_file.exists():
        try:
            with open(output_file, "rb") as f:
                print("✅ Loaded preprocessed documents from cache.")
                return pickle.load(f)
        except EOFError:
            print("⚠️ Cache file is corrupted or empty. Regenerating documents...")
            if output_file.exists():
                output_file.unlink()
        except Exception as e:
            print(f"⚠️ Error loading cache: {str(e)}. Regenerating documents...")

    docs = []
    for html_file, label in get_data_paths():
        docs.extend(extract_documents_from_html(html_file, label))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(docs, f)
    print("✅ Extracted and cached documents.")
    return docs


def main():
    if is_dev_mode():
        print("🚧 Running in development mode...")
        docs = load_or_generate_documents()
        print(f"🧾 Loaded {len(docs)} documents for processing.")

        vectorstore = create_or_load_vectorstore(docs)

        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=None)  # None will use env var
        graph = build_sdg_graph(docs, vectorstore, llm)
        initial_state = SDGState(input="How did LLMs evolve in 2023?")
        
        result = graph.invoke(initial_state)
        print("🧠 Agent Output:")
        print(json.dumps(result, indent=2, ensure_ascii=False, cls=DocumentEncoder))
    else:
        print("🔒 Production mode detected. Skipping document generation.")


if __name__ == "__main__":
    main()