from graph.types import SDGState
from graph.nodes.evolve import evolve_question
from unittest.mock import MagicMock

def test_evolve_question_modifies_state():
    state = SDGState(input="What were the top LLMs in 2023?")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Evolved: What were the top LLMs in 2023?")
    updated_state = evolve_question(state, mock_llm)

    assert updated_state.evolved_question.startswith("Evolved:")
    assert updated_state.evolved_question.endswith("2023?")