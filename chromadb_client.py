import chromadb
from chromadb.config import Settings
import time

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
print("client running on localhost:8000 -> http://localhost:8000")
