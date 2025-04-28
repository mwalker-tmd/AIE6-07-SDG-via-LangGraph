import streamlit as st
import json
import logging
import os
from preprocess.html_to_documents import extract_documents_from_html
from preprocess.embed_documents import create_or_load_vectorstore
from graph.build_graph import build_sdg_graph
from graph.types import SDGState
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    vectorstore_path = os.environ.get("VECTORSTORE_PATH", "/tmp/vectorstore")
    vectorstore = create_or_load_vectorstore(docs, path=vectorstore_path)
    
    # Initialize LLM client
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=None)  # None will use env var
    
    # Build graph
    graph = build_sdg_graph(docs, vectorstore, llm)
    
    st.success("Resources initialized successfully!")
    return docs, vectorstore, graph

# Initialize resources
docs, vectorstore, graph = initialize_resources()

# Add a number input for evolution passes
num_evolve_passes = st.number_input(
    label="Number of Evolution Passes",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
    help="How many times to evolve the question (alternates between challenging and creative prompts)."
)

# Generate synthetic data button
if st.button("Generate Synthetic Data"):
    with st.spinner("Generating synthetic data..."):
        # Create initial state
        state = SDGState(
            input="Generate synthetic data about LLM evolution",
            documents=[],
            evolved_questions=[],
            context=[],
            answer="",
            num_evolve_passes=num_evolve_passes
        )
        
        # Run the graph for each evolution pass
        all_results = []
        for i in range(num_evolve_passes):
            logger.debug(f"Running evolution pass {i+1}/{num_evolve_passes}")
            result = graph.invoke(state)
            if not isinstance(result, SDGState):
                result = SDGState(**dict(result))
            all_results.append(result)
            # Update state for next iteration with evolved questions
            state = SDGState(
                input=state.input,
                documents=state.documents,
                evolved_questions=result.evolved_questions,  # Pass forward all evolved questions
                context=[],  # Reset context for next iteration
                answer="",   # Reset answer for next iteration
                num_evolve_passes=num_evolve_passes
            )
        
        # Display results
        st.subheader("Generated Data")
        
        # Display evolved questions
        st.markdown("### Evolved Questions")
        evolved_questions = [
            {"id": f"q{i}", "question": result.evolved_questions[-1], "evolution_type": "simple"}
            for i, result in enumerate(all_results)
        ]
        st.json(evolved_questions)
        
        # Display answers
        st.markdown("### Answers")
        answers = [
            {"id": f"q{i}", "answer": result.answer}
            for i, result in enumerate(all_results)
        ]
        st.json(answers)
        
        # Display contexts
        st.markdown("### Contexts")
        contexts = [
            {"id": f"q{i}", "contexts": result.context}
            for i, result in enumerate(all_results)
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