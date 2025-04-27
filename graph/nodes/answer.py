from graph.types import SDGState
import logging

logger = logging.getLogger(__name__)

def generate_answer(state: SDGState) -> SDGState:
    """
    Synthesizes an answer from the retrieved context.
    This is a placeholder and would normally call an LLM in production.
    """
    logger.debug(f"Answer node received state: {state}")
    
    # Generate the answer
    context_snippet = "\n".join(state.context)
    
    # Create a new state with the generated answer
    new_state = SDGState(
        input=state.input,
        documents=state.documents,
        evolved_question=state.evolved_question,
        context=state.context,
        answer=f"Based on the retrieved context:\n{context_snippet}"
    )
    
    logger.debug(f"Answer node returning state: {new_state}")
    return new_state