from graph.types import SDGState
import logging

logger = logging.getLogger(__name__)

def retrieve_relevant_context(state: SDGState, vectorstore) -> SDGState:
    logger.debug(f"Retrieve node received state: {state}")
    
    # Perform retrieval
    retrieved_docs = vectorstore.similarity_search(state.evolved_question, k=5)
    logger.debug(f"Retrieved {len(retrieved_docs)} documents")
    
    # Create a new state with the retrieved context
    new_state = SDGState(
        input=state.input,
        documents=state.documents,
        evolved_question=state.evolved_question,
        context=[doc.page_content for doc in retrieved_docs],
        answer=state.answer
    )
    
    logger.debug(f"Retrieve node returning state: {new_state}")
    return new_state