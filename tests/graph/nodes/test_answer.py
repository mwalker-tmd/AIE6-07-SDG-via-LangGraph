from graph.types import SDGState
from graph.nodes.answer import generate_answer

def test_generate_answer_creates_string():
    state = SDGState(input="Test input", context=["Doc 1", "Doc 2"])
    updated = generate_answer(state)
    assert isinstance(updated.answer, str)
    assert "Doc 1" in updated.answer