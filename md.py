from args import get_args
from time import time
from typing import List
import os
import chromadb
from chromadb.config import Settings
import openai
import camelot
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()
args = get_args()
openai.api_key = os.getenv("OPENAI_API_KEY")
pdf_path="/Users/bappa123/Downloads/CUAD_v1/full_contract_pdf/Part_I/License_Agreements/AlliedEsportsEntertainmentInc_20190815_8-K_EX-10.19_11788293_EX-10.19_Content License Agreement.pdf"
pdf_name= str(str(pdf_path).split("/")[-1]).split(".")[0]
def custom_chunk_document_pdf(
    pdf_path: str, 
    chunk_size=500, 
    overlap=100
) -> List[str]:
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

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    final_text_chunks = []
    for chunk in text_chunks:
        final_text_chunks.extend(splitter.split_text(chunk))

    return final_text_chunks + table_chunks


chunks = custom_chunk_document_pdf(pdf_path)


md_output_path = f'/Users/bappa123/Desktop/agentic_rag/store/check_documents/{pdf_name}.md'

with open(md_output_path, 'w', encoding='utf-8') as md_file:
    for idx, chunk in enumerate(chunks):
        md_file.write(chunk + "\n")

print(f"{md_output_path} done ")


