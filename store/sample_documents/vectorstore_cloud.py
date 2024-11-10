from pinecone import Pinecone, Index, ServerlessSpec
import camelot.io as camelot
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import fitz  # PyMuPDF
import re
import json
import os


pc = Pinecone(api_key="f5371ada-dcde-45fe-8e21-42e08057a865")
index = pc.Index("pathway-ps-101")
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

try:
    with open("doc_index_mapping.json", "r") as f:
        doc_index_mapping = json.load(f)
except FileNotFoundError:
    doc_index_mapping = {}


def custom_chunk_document_pdf(pdf_path: str, chunk_size:int=500, overlap:int=100) -> List[str]:
    table_chunks = []
    text_chunks = []

    # Extract tables using Camelot
    # Use 'stream' or 'lattice' depending on table structure
    tables = camelot.read_pdf(pdf_path, pages='1-end', flavor='stream')
    for table in tables:
        df = table.df  # Get DataFrame of the table
        table_chunks.append(df.to_string(index=False))
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text_chunks.append(text.strip())

    # Chunk main text with RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    final_text_chunks = []
    for chunk in text_chunks:
        final_text_chunks.extend(splitter.split_text(chunk))

    return final_text_chunks + table_chunks


def get_namespace(pdf_path: str) -> str:
    file_name = os.path.basename(pdf_path)
    # Remove the file extension
    namespace = re.sub(r'\.pdf$', '', file_name, flags=re.IGNORECASE)
    # Replace invalid characters with underscores
    namespace = re.sub(r'[^a-zA-Z0-9_-]', '_', namespace)
    # Ensure it doesn't start or end with an invalid character
    namespace = namespace.strip('_-')
    # Limit to 63 characters
    if len(namespace) >= 63:
        namespace = namespace[:61]
        namespace = namespace + "A"

    # print(f"collection name:{collection_name}")
    return namespace


def get_index(namespace: str, pdfpath: str) -> Index:
    if namespace not in pc.list_indexes().names():
        pc.create_index(namespace, dimension=1536, metric='cosine', spec=spec)
        print(f"Created new index: {namespace}")
        doc_index_mapping[os.basename(pdfpath)] = namespace
    else:
        print(f"Index {namespace} already exists.")

    with open("doc_index_mapping.json", "w") as f:
        json.dump(doc_index_mapping, f)

    return pc.Index(namespace)


def populate_index(pdf_path: str, index: Index):
    chunks = custom_chunk_document_pdf(pdf_path)
    doc_id = get_namespace(pdf_path)
    
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


def make_index(pdf_path: str):
    namespace = get_namespace(pdf_path)
    index = get_index(namespace, pdf_path)
    populate_index(pdf_path, index)


def main():
    pdf_path = "CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf"
    make_index(pdf_path)


if __name__ == "__main__":
    main()   
