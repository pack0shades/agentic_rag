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
    # chroma_client.reset()
    # print ("client sleeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeping")
    openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key,
        model_name="text-embedding-ada-002"
    )
    # print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", chroma_client.list_collections())
    #print(f"ye rhi list {chroma_client.list_collections()}")
    collection = chroma_client.get_or_create_collection(
        name=filename, embedding_function=openai_ef)
    collection_list = []
    for i in range(len(chroma_client.list_collections())):
        collection_name = chroma_client.list_collections()[i].name
        #print(f"ye rha collection name:{collection_name}")
        collection_list.append(collection_name)
    return collection, collection_list


def retrieve_documents(collection, query: str, n_results=args.retrieved_docs) -> str:
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


def return_final_retrieve_docs(query):
    collection, collection_present = get_collection("SmartRxSystemsInc_20180914_1-A_EX1A-6_MAT_CTRCT_11351705_EX1AA")
    retrieved_docs = retrieve_documents(collection, query)
    return retrieved_docs


if __name__ == "__main__":
    query = "What are the benefits of using RAG for information retrieval?"
    start = time.time()
    results = return_final_retrieve_docs(query)
    end = time.time()
    print(results)
    # chroma_client = chromadb.HttpClient(host='localhost', port=5000)
    # chroma_client.reset()
    # chroma_client = chromadb.HttpClient(host='localhost', port=5000)
    # # chroma_client.reset()
    # # print ("client sleeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeping")
    # time.sleep(50)
    # openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    #     api_key=openai.api_key,
    #     model_name="text-embedding-ada-002"
    # )
    # print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", chroma_client.list_collections())
    # #print(f"ye rhi list {chroma_client.list_collections()}")
    # collection = chroma_client.get_or_create_collection(
    #     name="InnerscopeHearingTechnologiesInc_20181109_8-K_EX-10_6_1141970A", embedding_function=openai_ef)
    # print(collection.query(query_texts=["What are the benefits of using RAG for information retrieval?"], n_results=5))
