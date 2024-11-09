from agent import *
from chroma_db import *
import argparse
from chromadb_client import *
from reranker import *
from openai import OpenAI
from typing import List
from args import get_args
from swarm_router import get_agents, multi_agent
from agent import context_to_agent
import argparse

args = get_args()

# Print parsed arguments to verify correct parsing
print("Parsed arguments:", args)


def get_context(collection, reranker, query: str) -> str:
    # Retrieve documents
    retrieved_docs = retrieve_documents(collection, query)  # list
    if args.use_reranker:
        # Rerank documents
        reranked_docs = reranker.rerank_documents(query, retrieved_docs)
    else:
        reranked_docs = retrieved_docs

    context = ""
    for idx, doc in enumerate(reranked_docs):
        context += f"Rank {idx + 1}: {doc}\n"
    return context

def generate_response_from_context(quer: str, context)-> str:
    prompt = f'''Based on the following context, answer the user’s query accurately and concisely.

    Context: {context}
    User Query: {quer}
    
    Response:'''
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "you are a useful assistant"},
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ]
    )
    return (completion.choices[0].message.content)

def generate_response_from_multi_agent(query: str, context) -> str:
    agents, meta_agent, final_agent, router = get_agents()

    return multi_agent(agents, meta_agent, final_agent, router, query, context)


def pipeline(collection, reranker, query: str) -> str:
    context = get_context(collection, reranker, query)
    # print(f"ye rha context:::{context}")
    if args.pipeline == "nov4":
        final_context = context_to_agent(context)
        res = generate_response_from_context(query, final_context)
        return res
    elif args.pipeline == "nov9":
        res = generate_response_from_multi_agent(query, context)
        return res
    else:
        print("use --pipeline argument to specify the pipeline, available options are: 'nov4', 'nov9'")


def main():
    print(
        f"Using Reranker: {args.use_reranker}_____number of Retrieved Docs: {args.retrieved_docs}")
    collection, collection_list = get_collection('collection-1731096205')
    reranker_model = DocumentReranker()

    query = input("Enter your query: ")
    start_time = time.time()
    res = pipeline(collection, reranker_model, query)
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken:.4f} seconds")
    print(res)


if __name__ == "__main__":
    main()
