from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class ConfigurationSettings(BaseSettings):

    pdf_folder_path: Path = Field(description="Path to the folder containing the input PDF files")

    hf_token: str = Field(description="HuggingFace API token for accessing models")

    embeddings_model: str = Field(
        default="intfloat/multilingual-e5-base",
        description="HuggingFace model name for embeddings",
    )

    llm_model: str = Field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        description="HuggingFace model name for the LLM",
    )

    faiss_index_path: Path = Field(
        description="Path to store/load the FAISS index",
    )

    class Config:
        env_file = ".env"
