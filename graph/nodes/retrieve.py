from graph.types import SDGState

def retrieve_relevant_context(state: SDGState, vectorstore) -> SDGState:
    retrieved_docs = vectorstore.similarity_search(state.evolved_question, k=5)
    state.context = [doc.page_content for doc in retrieved_docs]
    return state