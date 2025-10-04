import torch
import logging

from typing import Tuple
from transformers import pipeline
from langchain.docstore.document import Document

MAX_TOKENS = 512

SYSTEM_PROMPT = """
Role: You are an insurance assistant answering questions about Athora pension products only using the provided document chunks.

Rules:
1. If the question is about a specific product (e.g., i-Pension, Nu Pension, Exclusief Pension), use only chunks from that product (see source in metadata).
2. If the question is general, you may use chunks from multiple products but always state which products you are referencing.
3. Always cite sources using Chunk IDs in double brackets (e.g., [[1]]).
4. Do not invent or assume information. If the answer cannot be found, reply exactly: “I’m sorry, I couldn’t find that information in our documents.”
5. Write in clear, simple, customer-friendly language. Avoid technical jargon.

Task: Answer the following user question using the provided chunks.
"""


USER_PROMPT_TEMPLATE = """
Question: {query}
Context: {relevant_content}
Answer:
"""


def init_pipeline(model_name: str, token: str):
   """
   Initialize the text generation pipeline with the specified model.

   Args:
      model_name (str): The name or path of the model to load.
      token (str): The authentication token for accessing the model.
    
   Returns:
      A text generation pipeline.
   """
   device = "cuda" if torch.cuda.is_available() else "cpu"
   logging.info(f"Using device: {device}")

   return pipeline(
        "text-generation",
        model=model_name,
        token=token,
        dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,  # <- Fix here
   )


def generate_answer(
    query: str, relevant_content: dict[str, dict[str,str]], pipe
) -> Tuple[str, dict[str, Document]]:
   """
   Generate an answer using LLM with references to chunks.
    
   Args:
      query (str): The user's question.
      relevant_content (dict): The relevant document chunks with their metadata.
      pipe: The text generation pipeline.

   Returns:
      Tuple[str, dict]: The generated answer and the references used.
   """

   user_prompt = USER_PROMPT_TEMPLATE.format(query=query, relevant_content=relevant_content)
   messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
   ]

   logging.info("Generating answer...")
   output = pipe(messages, max_new_tokens=MAX_TOKENS)
   answer = output[0]["generated_text"][-1]["content"]

   return answer
