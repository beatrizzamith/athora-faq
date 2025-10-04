import logging
import streamlit as st

from dotenv import load_dotenv
from time import perf_counter
from pathlib import Path

from athora_faq import (
    chunk_documents,
    build_faiss_index,
    load_faiss_index,
    build_references,
    init_pipeline,
    generate_answer,
    ConfigurationSettings,
)

load_dotenv(override=True)

TOP_K = 15
settings = ConfigurationSettings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Athora FAQ Assistant", page_icon="📘")
st.title("📘 Athora FAQ Assistant")
st.caption("AI-powered assistant for Athora’s pension product documentation.")


@st.cache_resource(show_spinner="🔧 Building or loading FAISS index...")
def get_index():
    """
    Build or load a FAISS index.
    This function is cached to avoid recomputation between Streamlit reruns.
    """
    index_path = Path(settings.faiss_index_path)
    pdf_folder = Path(settings.pdf_folder_path)

    if not index_path.exists():
        docs = chunk_documents(pdf_folder)
        logger.info("Loaded %d document chunks from %s", len(docs), pdf_folder)

        return build_faiss_index(docs, settings.embeddings_model, index_path)
    
    logger.info("Loading existing FAISS index from %s", index_path)              
    return load_faiss_index(index_path)


@st.cache_resource(show_spinner="🚀 Initializing language model pipeline...")
def get_pipeline():
    """Initialize the text generation pipeline, cached across reruns."""
    return init_pipeline(token=settings.hf_token, model_name=settings.llm_model)


# Initialize resources
index = get_index()
pipe = get_pipeline()

query = st.text_input("💬 Ask a question about Athora products:")

if query:
    with st.spinner("🔎 Generating answer... please wait..."):
        start = perf_counter()
        try:
            results = index.similarity_search(query, k=TOP_K)
            references = build_references(results)
            answer = generate_answer(query, references, pipe)

            # Display the generated answer
            st.subheader("✅ Answer")
            st.write(answer)

            # Display references in an expandable section
            with st.expander("📂 References used"):
                for ref_id, info in references.items():
                    st.markdown(f"**{ref_id}** – {info['source']}")
                    with st.expander("Show content"):
                        st.caption(info["content"])

            elapsed = perf_counter() - start
            logger.info("Answer generated in %.2f seconds", elapsed)

        except Exception as e:
            st.error("❌ An error occurred while generating the answer.")
            logger.exception("Error during answer generation: %s", e)