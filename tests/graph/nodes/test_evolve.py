from graph.types import SDGState
from graph.nodes.evolve import evolve_question
from unittest.mock import MagicMock, call

def test_evolve_question_modifies_state():
    state = SDGState(input="What were the top LLMs in 2023?")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Evolved: What were the top LLMs in 2023?")
    updated_state = evolve_question(state, mock_llm)

    assert updated_state.evolved_question.startswith("Evolved:")
    assert updated_state.evolved_question.endswith("2023?")

def test_evolve_question_multiple_passes():
    state = SDGState(input="What were the top LLMs in 2023?", num_evolve_passes=2)
    mock_llm = MagicMock()
    # Simulate different responses for each pass
    mock_llm.invoke.side_effect = [
        MagicMock(content="Challenging: What were the top LLMs in 2023?"),
        MagicMock(content="Creative: What were the top LLMs in 2023?")
    ]
    updated_state = evolve_question(state, mock_llm)
    # Should use the last evolution
    assert updated_state.evolved_question.startswith("Creative:")
    # Check that the LLM was called twice with alternating prompts
    expected_calls = [
        call("Rewrite or evolve the following question to be more challenging or insightful:\n\nWhat were the top LLMs in 2023?"),
        call("Rewrite or evolve the following question to be more creative or original:\n\nChallenging: What were the top LLMs in 2023?")
    ]
    mock_llm.invoke.assert_has_calls(expected_calls)

def test_evolve_question_single_pass():
    state = SDGState(input="What were the top LLMs in 2023?", num_evolve_passes=1)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Challenging: What were the top LLMs in 2023?")
    updated_state = evolve_question(state, mock_llm)
    assert updated_state.evolved_question.startswith("Challenging:")
    mock_llm.invoke.assert_called_once_with(
        "Rewrite or evolve the following question to be more challenging or insightful:\n\nWhat were the top LLMs in 2023?"
    )

def test_evolve_question_three_passes():
    state = SDGState(input="What were the top LLMs in 2023?", num_evolve_passes=3)
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content="Challenging: What were the top LLMs in 2023?"),
        MagicMock(content="Creative: What were the top LLMs in 2023?"),
        MagicMock(content="Challenging Again: What were the top LLMs in 2023?")
    ]
    updated_state = evolve_question(state, mock_llm)
    assert updated_state.evolved_question.startswith("Challenging Again:")
    expected_calls = [
        call("Rewrite or evolve the following question to be more challenging or insightful:\n\nWhat were the top LLMs in 2023?"),
        call("Rewrite or evolve the following question to be more creative or original:\n\nChallenging: What were the top LLMs in 2023?"),
        call("Rewrite or evolve the following question to be more challenging or insightful:\n\nCreative: What were the top LLMs in 2023?")
    ]
    mock_llm.invoke.assert_has_calls(expected_calls)

def test_evolved_questions_list_populated_correctly():
    state = SDGState(input="Base question", num_evolve_passes=3)
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content="Challenging: Base question"),
        MagicMock(content="Creative: Challenging: Base question"),
        MagicMock(content="Challenging Again: Creative: Challenging: Base question")
    ]
    updated_state = evolve_question(state, mock_llm)
    # The evolved_questions list should contain the initial input plus one entry per pass
    assert updated_state.evolved_questions == [
        "Base question",
        "Challenging: Base question",
        "Creative: Challenging: Base question",
        "Challenging Again: Creative: Challenging: Base question"
    ]
    # The property should return the last one
    assert updated_state.evolved_question == "Challenging Again: Creative: Challenging: Base question"

def test_evolved_questions_list_with_existing_evolutions():
    # If the state already has evolved_questions, it should continue from the last
    state = SDGState(input="Base question", evolved_questions=["Base question", "First evolution"], num_evolve_passes=2)
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content="Second evolution"),
        MagicMock(content="Third evolution")
    ]
    updated_state = evolve_question(state, mock_llm)
    assert updated_state.evolved_questions == [
        "Base question",
        "First evolution",
        "Second evolution",
        "Third evolution"
    ]
    assert updated_state.evolved_question == "Third evolution"