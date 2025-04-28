from graph.prompts import question_evolution_prompt
from graph.types import SDGState
import logging

logger = logging.getLogger(__name__)

def evolve_question(state: SDGState, llm) -> SDGState:
    prompts = [
        "Rewrite or evolve the following question to be more challenging or insightful:\n\n{}",
        "Rewrite or evolve the following question to be more creative or original:\n\n{}"
    ]
    
    # Choose prompt based on number of existing evolutions (even/odd)
    prompt_idx = len(state.evolved_questions) % len(prompts)
    prompt = prompts[prompt_idx].format(state.evolved_question)
    
    # Generate new evolution
    response = llm.invoke(prompt)
    evolved = response.content if hasattr(response, 'content') else str(response)
    
    # Create new state with appended evolution
    new_state = SDGState(
        input=state.input,
        documents=state.documents,
        evolved_questions=state.evolved_questions + [evolved],
        context=state.context,
        answer=state.answer,
        num_evolve_passes=state.num_evolve_passes
    )
    logger.debug(f"Evolve node returning state: {new_state}")
    return new_state