import pytest
from unittest.mock import patch, mock_open, MagicMock
from preprocess.embed_documents import create_or_load_vectorstore
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

@patch("preprocess.embed_documents.FAISS.from_documents")
@patch("preprocess.embed_documents.OpenAIEmbeddings")
@patch("preprocess.embed_documents.Path.exists", return_value=False)  # <- force creation
@patch("preprocess.embed_documents.open", new_callable=mock_open)    # <- prevent real file I/O
def test_create_vectorstore_when_not_cached(mock_open_file, mock_exists, mock_embed, mock_faiss):
    docs = [Document(page_content="test", metadata={"source": "unit"})]
    mock_vectorstore = MagicMock()
    mock_faiss.return_value = mock_vectorstore
    
    vectorstore = create_or_load_vectorstore(docs, path="tests/tmp/vectorstore.pkl")

    assert vectorstore == mock_vectorstore
    mock_faiss.assert_called_once()
    mock_vectorstore.save_local.assert_called_once()


@patch("preprocess.embed_documents.FAISS.load_local")
@patch("preprocess.embed_documents.Path.exists", return_value=True)
@patch("preprocess.embed_documents.open", new_callable=mock_open)
@patch("preprocess.embed_documents.OpenAIEmbeddings")
def test_load_existing_vectorstore(mock_embeddings, mock_open_file, mock_exists, mock_load_local):
    mock_vectorstore = MagicMock()
    mock_load_local.return_value = mock_vectorstore
    mock_embeddings_instance = MagicMock()
    mock_embeddings.return_value = mock_embeddings_instance

    result = create_or_load_vectorstore([], path="tests/tmp/vectorstore.pkl")
    
    assert result == mock_vectorstore
    mock_load_local.assert_called_once()
    mock_embeddings.assert_called_once()