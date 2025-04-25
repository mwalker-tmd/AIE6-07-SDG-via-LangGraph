from langgraph.graph import StateGraph
from graph.types import SDGState
from graph.nodes.evolve import evolve_question


def build_sdg_graph(docs) -> StateGraph:
    builder = StateGraph(SDGState)
    builder.add_node("evolve", evolve_question)
    builder.set_entry_point("evolve")
    return builder.compile()