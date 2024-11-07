import chromadb
from chromadb.config import Settings
import openai
import camelot 
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()
import os
from typing import List
from time import time

openai.api_key = os.getenv("OPENAI_API_KEY")


def custom_chunk_document_pdf(pdf_path: str, chunk_size=500, overlap=100)-> List[str]:
    table_chunks = []
    text_chunks = []
    
    # Extract tables using Camelot
    tables = camelot.read_pdf(pdf_path, pages='1-end', flavor='stream')  # Use 'stream' or 'lattice' depending on table structure
    for table in tables:
        df = table.df  # Get DataFrame of the table
        table_chunks.append(df.to_string(index=False)) 
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  
        text_chunks.append(text.strip())
    
    # Chunk main text with RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    final_text_chunks = []
    for chunk in text_chunks:
        final_text_chunks.extend(splitter.split_text(chunk))

    return final_text_chunks + table_chunks


def embed_and_store_chunks(doc_id, pdf_path: str, collection):
    """
    Embeds document chunks and stores them in Chroma DB.
    
    Args:
        doc_id (str): Unique identifier for the document.
        pdf_path (str): Path to the PDF document.
    """

    chunks = custom_chunk_document_pdf(pdf_path)

    for idx, chunk in enumerate(chunks):
        embedding = openai.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        ).data[0].embedding# Instead of generating embeddings manually, use the embedding function
        collection.add(
            ids=[f"{doc_id}_{idx}"],  # Unique ID for each chunk
            documents=[chunk],  # The document chunk to store
            embeddings=[embedding]  # The embedding of the chunk# Use the OpenAI embedding function
        )

    print(f"Stored {len(chunks)} chunks for document {doc_id}.")




if __name__ == "__main__":
        # Initialize Chroma DB through command chroma run --path /db_path
    pdf_path = './pdfs/nvidia.pdf'
    client = chromadb.HttpClient(host="localhost", port=8000)
    openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key,
        model_name="text-embedding-ada-002"
    )
    collection = client.get_or_create_collection(name="document_embeddings",
                                                 metadata={"hnsw:space": "cosine"},  # l2 is the default
                                                 embedding_function=openai_ef) 
    print("Chroma DB initialized.")
    print(f"{client.list_collections()}")

    embed_and_store_chunks("chromadb_doc", pdf_path, collection)
    print(f"Done!and here are some outputs\n{collection.peek()}\n\nthis is count::{collection.count()}")