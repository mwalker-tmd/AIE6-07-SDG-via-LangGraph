from graph.build_graph import build_sdg_graph
from graph.types import SDGState
from langchain.schema import Document

def test_graph_compiles_and_invokes():
    docs = [Document(page_content="A sample document", metadata={"source": "test", "filename": "test.html"})]
    graph = build_sdg_graph(docs)
    state = SDGState(input="Test question", documents=docs)
    result = graph.invoke(state)
    
    assert isinstance(result, dict)
    assert "evolved_question" in result
    assert result["evolved_question"].startswith("Evolved version of:")