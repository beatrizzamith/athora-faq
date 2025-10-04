import re
import logging

from pathlib import Path

from langchain.text_splitter import SpacyTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

CHUNK_SIZE = 300
CHUNK_OVERLAP = 60


def extract_text(pdf_path: Path) -> str:
    """
    Extract raw text from a PDF file using PyPDFLoader.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        str: Extracted text, or empty string if extraction fails.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.error(f"Failed to extract {pdf_path.name}: {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean up raw extracted text by removing symbols and normalizing whitespace.

    Args:
        text: Raw text extracted from a PDF.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"[■▪◾◽•◦⚪]", "", text)  # Remove bullets/symbols
    text = re.sub(r"[ \t]+", " ", text)  # Normalize spaces
    text = re.sub(r"\n{2,}", "\n", text)  # Collapse multiple newlines
    return text.strip()


def chunk_documents(pdf_folder: Path) -> list[Document]:
    """
    Extract, clean, and split all PDFs in a folder into LangChain Documents.

    Args:
        pdf_folder: Path to a folder containing PDF files.

    Returns:
        List[Document]: List of chunked and metadata-tagged LangChain Documents.
    """
    splitter = SpacyTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []

    for pdf_file in pdf_folder.glob("*.pdf"):
        logger.info(f"Processing {pdf_file.name}")
        text = extract_text(pdf_file)
        if not text:
            continue

        clean_txt = clean_text(text)

        split_text = splitter.split_text(clean_txt)

        for chunk in split_text:
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={"source": pdf_file.name},
                )
            )

    return chunks
