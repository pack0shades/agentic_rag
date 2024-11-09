import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from args import get_args

args = get_args()

class JinaReranker:
    def __init__(self, model_name="jinaai/jina-reranker-v2-base-multilingual"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True
        )
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def rerank_documents(self, query, documents, topk) -> list[str]:
        # Initialize list to store document scores
        scored_docs = []
        
        # Process each document
        for doc in documents:
            # Tokenize the query and document pair
            inputs = self.tokenizer(
                query, doc, padding=True, truncation=True, return_tensors="pt"
            ).to(self.model.device)
            
            # Get the relevance score from the model
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits[0].item()  # Assuming the model returns a single relevance score
            
            # Append document and score
            scored_docs.append((doc, score))
        
        # Sort documents by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the top-k documents
        top_documents = [doc for doc, _ in scored_docs[:topk]]
        
        return top_documents

from transformers import AutoTokenizer, AutoModel
import torch

class BAAIReranker:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def rerank_documents(self, query, documents, topk, instruction="为文章排序: ") -> list[str]:
        # Prepare query and documents embeddings
        all_texts = [instruction + query] + documents  # Instruction only added to query
        encoded_input = self.tokenizer(
            all_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # Compute embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]  # CLS pooling
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Separate query embedding from document embeddings
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]

        # Compute similarity scores (dot product) between query and each document
        scores = torch.matmul(doc_embeddings, query_embedding)
        
        # Sort documents by score in descending order
        scored_docs = [(doc, score.item()) for doc, score in zip(documents, scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the top-k documents
        top_documents = [doc for doc, _ in scored_docs[:topk]]
        
        return top_documents



# Example usage
if __name__ == "__main__":
    query = "Organic skincare products for sensitive skin"
    retrieved_docs = [
        "Organic skincare for sensitive skin with aloe vera and chamomile.",
        "i am laksh",
        "i am studing",
        "i am in 2nd year",
        "New makeup trends focus on bold colors and innovative techniques",
        "i am a cse student",
        "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille"
    ]
    
    reranker = BAAIReranker()
    reranked_docs = reranker.rerank_documents(query, retrieved_docs, 7)

    for idx, doc in enumerate(reranked_docs):
        print(f"Rank {idx + 1}: {doc}")
