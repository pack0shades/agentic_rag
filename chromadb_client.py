import os
import chromadb
from chromadb.config import Settings
import time
import openai
from args import get_args

args = get_args()

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_collection(filename='documentembeddings'):
    chroma_client = chromadb.HttpClient(host='localhost', port=5000)
    openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key,
        model_name="text-embedding-ada-002"
    )
    #print(f"ye rhi list {chroma_client.list_collections()}")
    collection = chroma_client.get_or_create_collection(
        name=filename, embedding_function=openai_ef)
    collection_list = []
    chroma_client.reset()
    print(f"deleted")
    time.sleep(10)
    for i in range(len(chroma_client.list_collections())):
        collection_name = chroma_client.list_collections()[i].name
        #print(f"ye rha collection name:{collection_name}")
        collection_list.append(collection_name)
    return collection, collection_list


def retrieve_documents(collection, query: str, n_results=args.retrieved_docs) ->list:
    results = collection.query(query_texts=[query], n_results=n_results)
    print(f"results in chor:::::::::{results}")
    return results["documents"][0]


def return_final_retrieve_docs(query):
    collection, collection_present = get_collection()
    retrieved_docs = retrieve_documents(collection, query)
    return retrieved_docs


if __name__ == "__main__":
    query = "What are the benefits of using RAG for information retrieval?"
    start = time.time()
    results = return_final_retrieve_docs(query)
    end = time.time()
    print(results)
