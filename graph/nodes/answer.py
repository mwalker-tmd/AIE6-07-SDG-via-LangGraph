from graph.types import SDGState


def generate_answer(state: SDGState) -> SDGState:
    # Placeholder for final answer generation
    state.answer = f"Answer based on: {state.context}"
    return state