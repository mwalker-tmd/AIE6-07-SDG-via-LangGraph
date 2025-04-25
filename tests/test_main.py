import os
import pytest
from unittest.mock import patch, mock_open
from langchain.schema import Document
from main import main

from main import is_dev_mode, get_data_paths, load_or_generate_documents

"""
Test the main functionality
"""

def test_is_dev_mode(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert is_dev_mode() is True

    monkeypatch.setenv("ENVIRONMENT", "production")
    assert is_dev_mode() is False


def test_get_data_paths():
    paths = get_data_paths()
    assert isinstance(paths, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in paths)


@pytest.mark.skipif(
    not os.path.exists("data/2023_llms.html"),
    reason="2023_llms.html not found"
)
@pytest.mark.skipif(
    not os.path.exists("data/2024_llms.html"),
    reason="2024_llms.html not found"
)
def test_load_or_generate_documents():
    docs = load_or_generate_documents()
    assert isinstance(docs, list)
    assert len(docs) > 0


@patch("main.extract_documents_from_html")
@patch("main.pickle.dump")
@patch("main.pickle.load", return_value=None)  # Prevents use of real pickle file
@patch("main.open", new_callable=mock_open)
@patch("main.Path.exists", return_value=False)  # Forces regeneration path
@patch("main.get_data_paths", return_value=[("test.html", "test")])  # Mock to return single path
def test_generate_documents_when_no_cache(mock_get_paths, mock_exists, mock_open_file, mock_load, mock_dump, mock_extract):
    mock_extract.return_value = [
        Document(page_content="doc1", metadata={"source": "test", "filename": "fake.html"}),
        Document(page_content="doc2", metadata={"source": "test", "filename": "fake.html"})
    ]

    docs = load_or_generate_documents()

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    mock_dump.assert_called_once()
    mock_extract.assert_called_once_with("test.html", "test")


@patch("main.extract_documents_from_html")
@patch("main.pickle.dump")
@patch("main.pickle.load", side_effect=EOFError)
@patch("main.open", new_callable=mock_open)
@patch("main.Path.exists", return_value=True)
@patch("main.get_data_paths", return_value=[("test.html", "test")])  # Mock to return single path
@patch("pathlib.Path.unlink")  # Add mock for unlink
def test_handle_corrupted_cache(mock_unlink, mock_get_paths, mock_exists, mock_open_file, mock_load, mock_dump, mock_extract):
    mock_extract.return_value = [
        Document(page_content="doc1", metadata={"source": "test", "filename": "fake.html"}),
        Document(page_content="doc2", metadata={"source": "test", "filename": "fake.html"})
    ]

    docs = load_or_generate_documents()

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    mock_dump.assert_called_once()
    mock_extract.assert_called_once_with("test.html", "test")
    mock_unlink.assert_called_once()  # Verify the corrupted file was deleted


@patch("main.extract_documents_from_html")
@patch("main.pickle.dump")
@patch("main.pickle.load", side_effect=ValueError("Invalid pickle data"))
@patch("main.open", new_callable=mock_open)
@patch("main.Path.exists", return_value=True)
@patch("main.get_data_paths", return_value=[("test.html", "test")])
def test_handle_general_exception(mock_get_paths, mock_exists, mock_open_file, mock_load, mock_dump, mock_extract):
    mock_extract.return_value = [
        Document(page_content="doc1", metadata={"source": "test", "filename": "fake.html"}),
        Document(page_content="doc2", metadata={"source": "test", "filename": "fake.html"})
    ]

    docs = load_or_generate_documents()

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    mock_dump.assert_called_once()
    mock_extract.assert_called_once_with("test.html", "test")


@patch("main.build_sdg_graph")
@patch("main.create_or_load_vectorstore")
@patch("main.load_or_generate_documents")
@patch("main.is_dev_mode", return_value=True)
def test_main_runs_dev_mode(mock_dev, mock_docs, mock_vectorstore, mock_graph):
    mock_docs.return_value = [Document(page_content="test", metadata={"source": "unit"})]
    mock_vectorstore.return_value = "mock_vectorstore"
    mock_graph.return_value.invoke.return_value = {"answer": "Sample result"}

    main()

    mock_docs.assert_called_once()
    mock_vectorstore.assert_called_once()
    mock_graph.return_value.invoke.assert_called_once()