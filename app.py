import streamlit as st
import json
from preprocess.html_to_documents import extract_documents_from_html
from preprocess.embed_documents import create_or_load_vectorstore
from graph.build_graph import build_sdg_graph
from graph.types import SDGState

# Page config
st.set_page_config(
    page_title="SDG via LangGraph",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 Synthetic Data Generation via LangGraph")
st.markdown("This app demonstrates the RAGAS Synthetic Data Generation steps using LangGraph.")

# Initialize the graph and documents (this would be done once at startup)
@st.cache_resource
def initialize_resources():
    st.info("Initializing resources... This may take a moment.")
    
    # Load documents
    docs = []
    for html_file, label in [
        ("data/2023_llms.html", "llm-2023"),
        ("data/2024_llms.html", "llm-2024"),
    ]:
        docs.extend(extract_documents_from_html(html_file, label))
    
    # Create vectorstore
    vectorstore = create_or_load_vectorstore(docs)
    
    # Build graph
    graph = build_sdg_graph(docs, vectorstore)
    
    st.success("Resources initialized successfully!")
    return docs, vectorstore, graph

# Initialize resources
docs, vectorstore, graph = initialize_resources()

# Generate synthetic data button
if st.button("Generate Synthetic Data"):
    with st.spinner("Generating synthetic data..."):
        # Create initial state
        initial_state = SDGState(input="Generate synthetic data about LLM evolution")
        
        # Invoke the graph
        result = graph.invoke(initial_state)
        
        # Display results
        st.subheader("Generated Data")
        
        # Display evolved questions
        st.markdown("### Evolved Questions")
        evolved_questions = [
            {"id": f"q{i}", "question": q, "evolution_type": "simple"} 
            for i, q in enumerate([result.evolved_question])  # Currently only one question
        ]
        st.json(evolved_questions)
        
        # Display answers
        st.markdown("### Answers")
        answers = [
            {"id": "q0", "answer": result.answer}
        ]
        st.json(answers)
        
        # Display contexts
        st.markdown("### Contexts")
        contexts = [
            {"id": "q0", "contexts": result.context}
        ]
        st.json(contexts)
        
        # Download results
        results = {
            "evolved_questions": evolved_questions,
            "answers": answers,
            "contexts": contexts
        }
        
        st.download_button(
            label="Download Results as JSON",
            data=json.dumps(results, indent=2),
            file_name="synthetic_data.json",
            mime="application/json"
        ) 