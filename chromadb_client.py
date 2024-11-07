import chromadb
from chromadb.config import Settings
import time
import openai
import os
from args import get_args

args = get_args()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_collection():
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key,
        model_name="text-embedding-ada-002"
    )
    collection = chroma_client.get_collection('document_embeddings', embedding_function=openai_ef)
    return collection

def retrieve_documents(collection, query: str, n_results=args.retrieved_docs)-> str:
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]

def main(query):
    retrieved_docs = retrieve_documents(get_collection(), query)
    return retrieved_docs

if __name__ == "__main__":
    query = "What are the benefits of using RAG for information retrieval?"
    start = time.time()
    results = main(query)
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
    print(results)
