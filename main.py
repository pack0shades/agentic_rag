from agent import *
from chroma_db import *
import argparse
from chromadb_client import *
from reranker import *
from openai import OpenAI
from typing import List
from time import time
from args import get_args
from swarm_router import get_agents, multi_agent

args = get_args()


def get_context(collection, reranker, query: str)-> str:
    # Retrieve documents
    retrieved_docs = retrieve_documents(collection, query) ## list
    if args.use_reranker:
    # Rerank documents
        reranked_docs = reranker.rerank_documents(query, retrieved_docs)

    context = ""
    for idx, doc in enumerate(reranked_docs):
        context += f"Rank {idx + 1}: {doc}\n"
    return context


def generate_response_from_context(query: str, context)-> str:
    agents, meta_agent, final_agent, router = get_agents()

    return multi_agent(agents, meta_agent, final_agent, router, query, context)
    

def pipeline(collection, reranker, query: str)-> str:
    context = get_context(collection, reranker, query)
    # print(f"ye rha context:::{context}")
    final_context = context_to_agent(context)
    res = generate_response_from_context(query, final_context)
    return res

def main():
    print(f"Using Reranker: {args.use_reranker}_____number of Retrieved Docs: {args.retrieved_docs}")
    collection = get_collection()
    reranker_model = DocumentReranker()

    query = input("Enter your query: ")
    start_time = time.time()
    res = pipeline(collection, reranker_model, query)
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken:.4f} seconds")
    print (res)

if __name__ == "__main__":
    main()