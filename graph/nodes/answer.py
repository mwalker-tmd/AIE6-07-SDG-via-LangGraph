from graph.types import SDGState

def generate_answer(state: SDGState) -> SDGState:
    """
    Synthesizes an answer from the retrieved context.
    This is a placeholder and would normally call an LLM in production.
    """
    context_snippet = "\n".join(state.context)
    state.answer = f"Based on the retrieved context:\n{context_snippet}"
    return state