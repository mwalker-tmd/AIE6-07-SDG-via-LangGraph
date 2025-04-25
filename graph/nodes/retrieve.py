from graph.types import SDGState


def retrieve_relevant_context(state: SDGState) -> SDGState:
    # Placeholder for embedding-based context retrieval (to be added later)
    state.context = ["Context doc 1", "Context doc 2"]
    return state