from langgraph.graph import StateGraph
from graph.types import SDGState
from graph.nodes.evolve import evolve_question
from graph.nodes.retrieve import retrieve_relevant_context
from graph.nodes.answer import generate_answer


def build_sdg_graph(docs, vectorstore, llm) -> StateGraph:
    # Create a new graph with our state type
    builder = StateGraph(SDGState)

    # Add nodes with explicit state handling
    builder.add_node("evolve", lambda state: evolve_question(state, llm))
    builder.add_node("retrieve", lambda state: retrieve_relevant_context(state, vectorstore))
    builder.add_node("generate_answer", generate_answer)

    # Define the flow
    builder.set_entry_point("evolve")
    builder.add_edge("evolve", "retrieve")
    builder.add_edge("retrieve", "generate_answer")
    builder.set_finish_point("generate_answer")

    # Compile the graph
    graph = builder.compile()
    
    # Return the graph
    return graph