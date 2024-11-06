from agent import *
from chroma_db import *
import argparse
from chromadb_client import *
from reranker import *
from openai import OpenAI
from typing import List

def get_context(collection, reranker, query: str):
    # Retrieve documents
    retrieved_docs = retrieve_documents(collection, query) ## list
    # Rerank documents
    reranked_docs = reranker.rerank_documents(query, retrieved_docs)

    context = "";
    for idx, doc in enumerate(reranked_docs):
        context += f"Rank {idx + 1}: {doc}\n"
    return context


def generate_response_from_context(quer, context):
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
    

def pipeline(collection, reranker, query):
    context = get_context(collection, reranker, query)
    # print(f"ye rha context:::{context}")
    final_context = context_to_agent(context)
    res = generate_response_from_context(query, final_context)
    return res

def main():
    collection = get_collection()
    reranker_model = DocumentReranker()
    query = input("Enter your query: ")
    res = pipeline(collection, reranker_model, query)
    print (res)

if __name__ == "__main__":
    main()