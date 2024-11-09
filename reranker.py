import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from args import get_args

args = get_args()

class DocumentReranker:
    def __init__(self, model_name="jinaai/jina-reranker-v2-base-multilingual"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True
        )
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def rerank_documents(self, query, documents, topk) -> list[str]:
        # Construct sentence pairs with the query and each document
        sentence_pairs = [[query, doc] for doc in documents]
        
        # Tokenize and score each sentence pair
        scores = []
        for pair in sentence_pairs:
            inputs = self.tokenizer(
                pair[0], pair[1],
                return_tensors='pt',
                truncation=True,
                padding=True,
            )
            inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                score = torch.softmax(logits, dim=-1).item()  # Relevance score
                scores.append(score)

        # Sort documents based on scores
        reranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked_docs][:topk]


# Example usage
if __name__ == "__main__":
    query = "Organic skincare products for sensitive skin"
    retrieved_docs = [
        "Organic skincare for sensitive skin with aloe vera and chamomile.",
        "New makeup trends focus on bold colors and innovative techniques",
        "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
        "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
        "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
        "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
        "针对敏感肌专门设计的天然有机护肤产品",
        "新的化妆趋势注重鲜艳的颜色和创新的技巧",
        "敏感肌のために特別に設計された天然有機スキンケア製品",
        "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
    ]
    
    reranker = DocumentReranker()
    reranked_docs = reranker.rerank_documents(query, retrieved_docs, 6)

    for idx, doc in enumerate(reranked_docs):
        print(f"Rank {idx + 1}: {doc}")
