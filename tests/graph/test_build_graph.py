from graph.build_graph import build_sdg_graph
from graph.types import SDGState
from langchain.schema import Document
from unittest.mock import MagicMock

def test_build_sdg_graph_runs():
    docs = [Document(page_content="Sample content", metadata={"source": "test", "filename": "test.html"})]
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = [
        Document(page_content="Relevant content", metadata={})
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Evolved test question")

    graph = build_sdg_graph(docs, mock_vectorstore, mock_llm)
    state = SDGState(input="Test input", documents=docs)
    result = graph.invoke(state)

    assert isinstance(result, dict)
    assert "evolved_question" in result
    assert result["context"]
    assert "Relevant content" in result["context"][0]