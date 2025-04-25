from graph.prompts import question_evolution_prompt
from graph.types import SDGState


def evolve_question(state: SDGState) -> SDGState:
    # Placeholder for LLM-driven evolution
    state.evolved_question = f"Evolved version of: {state.input}"
    return state