import pathway as pw
from pathway.xpacks.llm.embedders import OpenAIEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer
import os
import time
from pathway.xpacks.llm.vector_store import VectorStoreClient

data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./check_documents",
        format="binary",
        mode="streaming",
        with_metadata=True
    )
)
print("here is it :")
print(data_sources)


PATHWAY_PORT = 8765

# Choose document transformers
text_splitter = TokenCountSplitter()
embedder = OpenAIEmbedder(
    api_key="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", 
    model="text-embedding-3-small"
)

vector_server = VectorStoreServer(
    *data_sources,
    embedder=embedder,
    splitter=text_splitter,
)

vector_server.run_server(
    host="127.0.0.1", 
    port=PATHWAY_PORT, 
    threaded=True, 
    with_cache=False
)

time.sleep(30)