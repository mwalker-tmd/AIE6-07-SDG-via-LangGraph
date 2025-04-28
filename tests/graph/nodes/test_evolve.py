from graph.types import SDGState
from graph.nodes.evolve import evolve_question
from unittest.mock import MagicMock, call

def test_evolve_question_initial_state():
    # Test evolution from initial state (should use input)
    state = SDGState(input="What were the top LLMs in 2023?")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Evolved: What were the top LLMs in 2023?")
    updated_state = evolve_question(state, mock_llm)

    # Should use challenging prompt first (even index)
    mock_llm.invoke.assert_called_once_with(
        "Rewrite or evolve the following question to be more challenging or insightful:\n\nWhat were the top LLMs in 2023?"
    )
    assert len(updated_state.evolved_questions) == 1
    assert updated_state.evolved_questions[0] == "Evolved: What were the top LLMs in 2023?"
    assert updated_state.evolved_question == "Evolved: What were the top LLMs in 2023?"

def test_evolve_question_with_one_evolution():
    # Test evolution with one existing evolution (should use creative prompt)
    state = SDGState(
        input="Base question",
        evolved_questions=["First evolution"]
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Creative evolution")
    updated_state = evolve_question(state, mock_llm)

    # Should use creative prompt (odd index)
    mock_llm.invoke.assert_called_once_with(
        "Rewrite or evolve the following question to be more creative or original:\n\nFirst evolution"
    )
    assert len(updated_state.evolved_questions) == 2
    assert updated_state.evolved_questions == ["First evolution", "Creative evolution"]
    assert updated_state.evolved_question == "Creative evolution"

def test_evolve_question_with_two_evolutions():
    # Test evolution with two existing evolutions (should use challenging prompt)
    state = SDGState(
        input="Base question",
        evolved_questions=["First evolution", "Second evolution"]
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Challenging evolution")
    updated_state = evolve_question(state, mock_llm)

    # Should use challenging prompt (even index)
    mock_llm.invoke.assert_called_once_with(
        "Rewrite or evolve the following question to be more challenging or insightful:\n\nSecond evolution"
    )
    assert len(updated_state.evolved_questions) == 3
    assert updated_state.evolved_questions == ["First evolution", "Second evolution", "Challenging evolution"]
    assert updated_state.evolved_question == "Challenging evolution"

def test_state_preservation():
    # Test that other state fields are preserved
    initial_state = SDGState(
        input="Base question",
        evolved_questions=["First evolution"],
        documents=[],
        context=["Some context"],
        answer="Previous answer",
        num_evolve_passes=5
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="New evolution")
    updated_state = evolve_question(initial_state, mock_llm)

    # Check that all fields are preserved except evolved_questions
    assert updated_state.input == initial_state.input
    assert updated_state.documents == initial_state.documents
    assert updated_state.context == initial_state.context
    assert updated_state.answer == initial_state.answer
    assert updated_state.num_evolve_passes == initial_state.num_evolve_passes
    # Check that evolved_questions is updated correctly
    assert len(updated_state.evolved_questions) == 2
    assert updated_state.evolved_questions[0] == "First evolution"
    assert updated_state.evolved_questions[1] == "New evolution"