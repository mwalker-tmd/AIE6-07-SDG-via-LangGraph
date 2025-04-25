import pytest
from unittest.mock import patch, mock_open
from preprocess.embed_documents import create_or_load_vectorstore
from langchain.schema import Document
from pathlib import Path

@patch("preprocess.embed_documents.pickle.dump")
@patch("preprocess.embed_documents.FAISS.from_documents")
@patch("preprocess.embed_documents.OpenAIEmbeddings")
@patch("preprocess.embed_documents.Path.exists", return_value=False)  # <- force creation
@patch("preprocess.embed_documents.open", new_callable=mock_open)    # <- prevent real file I/O
def test_create_vectorstore_when_not_cached(mock_open_file, mock_exists, mock_embed, mock_faiss, mock_pickle):
    docs = [Document(page_content="test", metadata={"source": "unit"})]
    mock_faiss.return_value = "fake_vectorstore"
    vectorstore = create_or_load_vectorstore(docs, path="tests/tmp/vectorstore.pkl")

    assert vectorstore == "fake_vectorstore"
    mock_faiss.assert_called_once()
    mock_pickle.assert_called_once()


@patch("preprocess.embed_documents.pickle.load", return_value="cached_vectorstore")
@patch("preprocess.embed_documents.Path.exists", return_value=True)
@patch("preprocess.embed_documents.open", new_callable=mock_open)
def test_load_existing_vectorstore(mock_open_file, mock_exists, mock_load):
    result = create_or_load_vectorstore([], path="tests/tmp/vectorstore.pkl")
    assert result == "cached_vectorstore"
    mock_load.assert_called_once()