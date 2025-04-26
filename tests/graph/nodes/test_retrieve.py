from graph.types import SDGState
from graph.nodes.retrieve import retrieve_relevant_context
from langchain.schema import Document
from unittest.mock import MagicMock

def test_retrieve_context_sets_list():
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = [
        Document(page_content="Relevant document 1", metadata={}),
        Document(page_content="Relevant document 2", metadata={}),
    ]

    state = SDGState(input="Test question", evolved_question="Test question evolved")
    updated = retrieve_relevant_context(state, mock_vectorstore)

    assert isinstance(updated.context, list)
    assert len(updated.context) == 2
    assert "Relevant document 1" in updated.context[0]