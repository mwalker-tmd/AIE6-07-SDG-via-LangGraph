from graph.types import SDGState
from graph.nodes.retrieve import retrieve_relevant_context

def test_retrieve_context_sets_list():
    state = SDGState(input="What about LLM evolution?")
    updated = retrieve_relevant_context(state)
    assert isinstance(updated.context, list)
    assert len(updated.context) > 0
