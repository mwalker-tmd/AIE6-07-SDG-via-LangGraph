from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import json
import numpy as np
import os


def create_or_load_vectorstore(docs: list[Document], path: str = "generated/vectorstore") -> FAISS:
    path = Path(path)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    if path.exists():
        print("✅ Loading FAISS VectorStore from disk...")
        vectorstore = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
        print("✅ Loaded FAISS VectorStore successfully.")
        return vectorstore

    print("Creating new FAISS VectorStore...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))
    print("✅ Created and cached FAISS VectorStore.")
    return vectorstore
