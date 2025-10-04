import logging

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

EMBEDDINGS_MODEL = "intfloat/multilingual-e5-base"

logger = logging.getLogger(__name__)


def build_faiss_index(docs: list[Document], embeddings_model: str, index_path: Path) -> FAISS:
    """
    Build and save a FAISS index from a list of documents.

    Args:
        docs: List of LangChain Document objects.
        embeddings_model: Name of HuggingFace embeddings model.
        index_path: Path where the FAISS index will be saved.

    Returns:
        FAISS: The built FAISS index.
    """
    logger.info("Building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    index = FAISS.from_documents(docs, embeddings)
    index.save_local(index_path)
    logger.info(f"FAISS index saved to {index_path}")

    return index


def build_references(retrieved_docs: list[Document]) -> dict[str, dict[str, str]]:
    """Create a mapping of reference IDs to document metadata and content."""
    references = {}
    for i, doc in enumerate(retrieved_docs, start=1):
        ref_id = f"[[{i}]]"
        references[ref_id] = {
            "source": doc.metadata["source"],
            "content": doc.page_content,
        }
    return references


def load_faiss_index(index_path: Path) -> FAISS:
    """
    Load a FAISS index from disk.

    Args:
        index_path: Path to the saved FAISS index.

    Returns:
        FAISS: The loaded FAISS index.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    logger.info(f"Loading FAISS index from {index_path}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return faiss_index
