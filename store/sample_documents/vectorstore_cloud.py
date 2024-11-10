from pinecone import Pinecone, Index, ServerlessSpec
import camelot.io as camelot
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import fitz  # PyMuPDF
from tqdm import tqdm
import re
import json
import os


api_key = os.getenv("PINECONE_API_KEY", "your_api_key_here")
region = os.getenv("PINECONE_REGION", "us-east-1")
cloud = os.getenv("PINECONE_CLOUD", "aws")

pc = Pinecone(api_key=api_key, environment=region)
spec = ServerlessSpec(cloud=cloud, region=region)


try:
    with open("doc_index_mapping.json", "r") as f:
        doc_index_mapping = json.load(f)
except FileNotFoundError:
    doc_index_mapping = {}


def custom_chunk_document_pdf(pdf_path: str, chunk_size:int=500, overlap:int=100) -> List[str]:
    table_chunks = []
    text_chunks = []

    tables = camelot.read_pdf(pdf_path, pages='1-end', flavor='stream')
    for table in tables:
        df = table.df 
        table_chunks.append(df.to_string(index=False))

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text_chunks.append(text.strip())

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    final_text_chunks = []
    for chunk in text_chunks:
        final_text_chunks.extend(splitter.split_text(chunk))

    return final_text_chunks + table_chunks


def get_namespace(pdf_path: str) -> str:
    file_name = os.path.basename(pdf_path)

    namespace = re.sub(r'\.pdf$', '', file_name, flags=re.IGNORECASE)
    namespace = re.sub(r'[^a-z0-9-]', '-', namespace.lower()).strip('-')
    if len(namespace) > 43:
        namespace = namespace[:43].rstrip('-')
    
    return namespace


def get_index(namespace: str, pdf_path: str) -> Index:
    if namespace not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(namespace, dimension=1536, metric='cosine', spec=spec)
        print(f"Created new index: {namespace}")
        doc_index_mapping[os.path.basename(pdf_path)] = namespace
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

        vectors.append(
            {
                "id": f"{doc_id}_{idx}", 
                "values": embedding,
                "metadata": {
                    "text": chunk
                }
            }
        )
    
    index.upsert(vectors=vectors, namespace=doc_id)
    print(f"Stored {len(chunks)} chunks for document {doc_id}.")

def make_index(pdf_path: str):
    namespace = get_namespace(pdf_path)
    index = get_index(namespace, pdf_path)
    populate_index(pdf_path, index)

def main():
    pdffilepaths=[
        os.path.join(root, file)
        for root, _, files in os.walk("./CUAD_v1")
        for file in files if file.endswith(".pdf")
    ]
    
    pdffilepaths += [
        os.path.join(root, file)
        for root, _, files in os.walk("./CUAD_v1")
        for file in files if file.endswith(".PDF")
    ]


    print (f"total pdfs : {len(pdffilepaths)}")

    for pdf_files in tqdm(pdffilepaths, desc="pdf to pinecone"):
        make_index(pdf_files)


if __name__ == "__main__":
    main()
