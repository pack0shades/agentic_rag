from pathway.xpacks.llm.vector_store import VectorStoreClient

client = VectorStoreClient(
    host="0.0.0.0",
    port=8765,
)

query = client(
    "What is india?"
)

print(query)