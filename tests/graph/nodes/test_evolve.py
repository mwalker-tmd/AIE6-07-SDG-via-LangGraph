from graph.types import SDGState
from graph.nodes.evolve import evolve_question

def test_evolve_question_modifies_state():
    state = SDGState(input="What were the top LLMs in 2023?")
    updated_state = evolve_question(state)

    assert updated_state.evolved_question.startswith("Evolved version of: ")
    assert updated_state.evolved_question.endswith("2023?")