from pinecone import Pinecone, Index

pc = Pinecone(api_key="f5371ada-dcde-45fe-8e21-42e08057a865")
index = pc.Index("pathway-ps-101")


def populate_index(pdf_path: str, index: Index):
    pass
