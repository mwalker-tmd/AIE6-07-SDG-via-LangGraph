from graph.prompts import question_evolution_prompt
from graph.types import SDGState
import logging

logger = logging.getLogger(__name__)

def evolve_question(state: SDGState, llm) -> SDGState:
    logger.debug(f"Evolve node received state: {state}")
    
    # Use the LLM to generate an evolved question
    prompt = f"Rewrite or evolve the following question to be more challenging or insightful:\n\n{state.input}"
    response = llm.invoke(prompt)
    evolved_question = response.content if hasattr(response, 'content') else str(response)

    new_state = SDGState(
        input=state.input,
        documents=state.documents,
        evolved_question=evolved_question,
        context=state.context,
        answer=state.answer
    )
    
    logger.debug(f"Evolve node returning state: {new_state}")
    return new_state