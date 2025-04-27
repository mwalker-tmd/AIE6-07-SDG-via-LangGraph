from graph.prompts import question_evolution_prompt
from graph.types import SDGState
import logging

logger = logging.getLogger(__name__)

def evolve_question(state: SDGState) -> SDGState:
    logger.debug(f"Evolve node received state: {state}")
    
    # Create a new state with the evolved question
    new_state = SDGState(
        input=state.input,
        documents=state.documents,
        evolved_question=f"Evolved version of: {state.input}",
        context=state.context,
        answer=state.answer
    )
    
    logger.debug(f"Evolve node returning state: {new_state}")
    return new_state