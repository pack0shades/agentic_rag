from pathway.xpacks.llm.vector_store import VectorStoreClient

client = VectorStoreClient(
    host="127.0.0.1",
    port=8765,
)

query = client(
    "What is Pathway??"
)

print(query)