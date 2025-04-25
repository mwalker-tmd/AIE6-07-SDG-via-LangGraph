from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from pathlib import Path
import pickle


def create_or_load_vectorstore(docs: list[Document], path: str = "generated/vectorstore.pkl") -> FAISS:
    path = Path(path)
    if path.exists():
        print("✅ Loaded FAISS VectorStore from disk.")
        with open(path, "rb") as f:
            return pickle.load(f)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorstore, f)
    print("✅ Created and cached FAISS VectorStore.")
    return vectorstore
