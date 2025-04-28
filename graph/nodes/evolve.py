from graph.prompts import question_evolution_prompt
from graph.types import SDGState
import logging

logger = logging.getLogger(__name__)

def evolve_question(state: SDGState, llm) -> SDGState:
    num_passes = getattr(state, "num_evolve_passes", 2)
    prompts = [
        "Rewrite or evolve the following question to be more challenging or insightful:\n\n{}",
        "Rewrite or evolve the following question to be more creative or original:\n\n{}"
    ]
    evolved_questions = list(state.evolved_questions) if state.evolved_questions else [state.input]
    evolved = evolved_questions[-1]
    for i in range(num_passes):
        prompt = prompts[i % len(prompts)].format(evolved)
        response = llm.invoke(prompt)
        evolved = response.content if hasattr(response, 'content') else str(response)
        evolved_questions.append(evolved)
    new_state = SDGState(
        input=state.input,
        documents=state.documents,
        evolved_questions=evolved_questions,
        context=state.context,
        answer=state.answer,
        num_evolve_passes=state.num_evolve_passes
    )
    logger.debug(f"Evolve node returning state: {new_state}")
    return new_state