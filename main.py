import os
import pickle
from preprocess.html_to_documents import extract_documents_from_html
from langchain.schema import Document
from pathlib import Path


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
        # Place LangGraph invocation here
    else:
        print("🔒 Production mode detected. Skipping document generation.")


if __name__ == "__main__":
    main()