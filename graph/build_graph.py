from langgraph.graph import StateGraph
from graph.types import SDGState
from graph.nodes.evolve import evolve_question
from graph.nodes.retrieve import retrieve_relevant_context
from graph.nodes.answer import generate_answer


def build_sdg_graph(docs, vectorstore) -> StateGraph:
    builder = StateGraph(SDGState)

    # Add nodes
    builder.add_node("evolve", evolve_question)
    builder.add_node("retrieve", lambda state: retrieve_relevant_context(state, vectorstore))
    builder.add_node("generate_answer", generate_answer)

    # Define flow
    builder.set_entry_point("evolve")
    builder.add_edge("evolve", "retrieve")
    builder.add_edge("retrieve", "generate_answer")

    return builder.compile()