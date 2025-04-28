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
                "evolved_questions": obj.evolved_questions,
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


def format_results(all_results):
    """Format results into the standard JSON structure."""
    evolved_questions = [
        {"id": f"q{i}", "question": result.evolved_questions[-1], "evolution_type": "simple"}
        for i, result in enumerate(all_results)
    ]
    answers = [
        {"id": f"q{i}", "answer": result.answer}
        for i, result in enumerate(all_results)
    ]
    contexts = [
        {"id": f"q{i}", "contexts": result.context}
        for i, result in enumerate(all_results)
    ]
    return {
        "evolved_questions": evolved_questions,
        "answers": answers,
        "contexts": contexts
    }


def main():
    if is_dev_mode():
        print("🚧 Running in development mode...")
        docs = load_or_generate_documents()
        print(f"🧾 Loaded {len(docs)} documents for processing.")

        vectorstore_path = os.environ.get("VECTORSTORE_PATH", "/tmp/vectorstore")
        vectorstore = create_or_load_vectorstore(docs, path=vectorstore_path)

        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=None)  # None will use env var
        graph = build_sdg_graph(docs, vectorstore, llm)
        
        # Set up initial state with desired number of passes
        num_evolve_passes = int(os.environ.get("NUM_EVOLVE_PASSES", "2"))
        state = SDGState(
            input="How did LLMs evolve in 2023?",
            documents=[],
            evolved_questions=[],
            context=[],
            answer="",
            num_evolve_passes=num_evolve_passes
        )
        
        # Run the graph for each evolution pass
        all_results = []
        print(f"🔄 Running {num_evolve_passes} evolution passes...")
        for i in range(num_evolve_passes):
            print(f"\n📝 Evolution pass {i+1}/{num_evolve_passes}:")
            result = graph.invoke(state)
            if not isinstance(result, SDGState):
                result = SDGState(**dict(result))
            all_results.append(result)
            # Update state for next iteration with evolved questions
            state = SDGState(
                input=state.input,
                documents=state.documents,
                evolved_questions=result.evolved_questions,  # Pass forward all evolved questions
                context=[],  # Reset context for next iteration
                answer="",   # Reset answer for next iteration
                num_evolve_passes=num_evolve_passes
            )
            print(f"  Question: {result.evolved_questions[-1]}")
            print(f"  Answer: {result.answer[:100]}...")
        
        # Format and output results
        print("\n🧠 Final Output:")
        results = format_results(all_results)
        print(json.dumps(results, indent=2, ensure_ascii=False, cls=DocumentEncoder))
        
        # Save results to file
        output_file = Path("generated/results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=DocumentEncoder)
        print(f"\n💾 Results saved to {output_file}")
    else:
        print("🔒 Production mode detected. Skipping document generation.")


if __name__ == "__main__":
    main()