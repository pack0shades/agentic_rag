from pinecone import Pinecone
from pinecone import ServerlessSpec
import openai
import camelot
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()
import os
from typing import List
import json
from time import time

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load existing mappings or create an empty dictionary
try:
    with open("doc_index_mapping.json", "r") as f:
        doc_index_mapping = json.load(f)
except FileNotFoundError:
    doc_index_mapping = {}

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# configure client
pc = Pinecone(api_key=pinecone_api_key)


cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)



def custom_chunk_document_pdf(pdf_path: str, chunk_size=500, overlap=100) -> List[str]:
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

def create_new_collection_for_pdf(doc_id: str):
    """
    Creates a new Pinecone index for a specific PDF document.
    
    Args:
        doc_id (str): Unique identifier for the document.
    
    Returns:
        pinecone.Index: The Pinecone index created for the document.
    """
    index_name = f"collection-{doc_id}"  # Create a unique name for each PDF collection
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension=1536, metric='cosine',spec=spec)
        print(f"Created new index: {index_name}")
        doc_index_mapping[doc_id] = index_name
    else:
        print(f"Index {index_name} already exists.")
    
    
    with open("doc_index_mapping.json", "w") as f:
        json.dump(doc_index_mapping, f)
        
    return pc.Index(index_name)



def embed_and_store_chunks(doc_id, pdf_path: str):
    """
    Embeds document chunks and stores them in a new Pinecone collection.
    
    Args:
        doc_id (str): Unique identifier for the document.
        pdf_path (str): Path to the PDF document.
    """
    index = create_new_collection_for_pdf(doc_id)
    chunks = custom_chunk_document_pdf(pdf_path)

    vectors = []
    for idx, chunk in enumerate(chunks):
        embedding = openai.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        ).data[0].embedding

        vector_id = f"{doc_id}_{idx}"
        vectors.append((vector_id, embedding, {"text": chunk}))

    # Upsert vectors to Pinecone
    index.upsert(vectors)
    print(f"Stored {len(chunks)} chunks for document {doc_id}.")

if __name__ == "__main__":
    pdf_path = './pdfs/nvidia.pdf'
    doc_id = input("Enter document ID(write generate to autogenerate): ")
    if doc_id == "generate":
        doc_id = f"doc-{int(time())}"  
    
    print("Pinecone initialized.")
    start_time = time()
    print(f"Existing indexes: {pc.list_indexes().names()}")

    embed_and_store_chunks(doc_id, pdf_path)
    print("Done!")
    time_taken = time() - start_time
    print(f"Time taken: {time_taken:.4f/60} minutes")

    index = pc.Index(f"collection-{doc_id}")
    stats = index.describe_index_stats()
    print(f"Index stats for {doc_id}: {stats}")
