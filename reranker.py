import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from args import get_args

args = get_args()

model_name="distilbert-base-uncased-finetuned-sst-2-english"
class DocumentReranker:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank_documents(self, query, documents)-> list[str]:
        
        query_encoding = self.tokenizer(query, return_tensors='pt')
        scores = []

        for doc in documents:
            inputs = self.tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512)
            
            combined_inputs = {**query_encoding, **inputs}
            
            with torch.no_grad():
                logits = self.model(**combined_inputs).logits
                score = torch.softmax(logits, dim=-1)[:, 1].item()  # Probability of relevance
                scores.append(score)
        
        reranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in reranked_docs]

# Example usage
if __name__ == "__main__":
    query = "What are the benefits of using RAG for information retrieval?"
    retrieved_docs = [
        "RAG is useful because it combines retrieval with generation.",
        "The retrieval step enhances the answer generation.",
        "Using RAG can improve answer accuracy by providing context.",
        "There are many uses of RAG, including summarization and question answering."
    ]
    reranker = DocumentReranker()

    reranked_docs = reranker.rerank_documents(query, retrieved_docs)

    for idx, doc in enumerate(reranked_docs):
        print(f"Rank {idx + 1}: {doc}")
