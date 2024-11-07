import openai
import pandas as pd
import os
from args import get_args
from main import *
from time import time

args = get_args()

def judge_llm(ground_truth, generated_answer):
    prompt = f"""
    Ground Truth: "{ground_truth}"
    Generated Answer: "{generated_answer}"
    Evaluate the similarity of two given text snippets. Input: Ground Truth Text, Predicted Answer Text.
    Output: 1 if the Predicted Answer Text has the same context as the Ground Truth Text, 0 otherwise 
    '''do not answer anything except 0 or 1'''.
    """

    
    response = openai.Completion.create(
        model="gpt-40-mini",
        prompt=prompt,
        max_tokens=1,
        temperature=0
    )

    output = response.choices[0].text.strip()
    return int(output) if output in {"1", "0"} else 0

def find_pdf(data_dir: str, filename: str)->str:
    filename = filename + ".pdf"
    for dirpath, part,dirnames, pdfs in os.walk(data_dir):
        for file in pdfs:
            if file == filename:
                return os.path.join(dirpath, file)
            else:
                print(f"File {filename} not found in {dirpath}.AAAAAAAAAAAaaaaaaaaaaaaaAAAAAAAAAAAAAAAAA")
                return None
            


def main():
    print(f"Using Reranker: {args.use_reranker}_____number of Retrieved Docs: {args.retrieved_docs}")
    df = pd.read_csv("cuad_qas_with_responces.csv")
    start_time = time()

    for index, row in df.iterrows():
        query = row['question']
        print(f"question:{query}")
        filename = row['id']
        pdf_loc = find_pdf(filename)

        collection, collection_present = get_collection(filename)
        if not collection_present:
            embed_and_store_chunks(pdf_path=pdf_loc,doc_id=filename, collection=collection)
            print(f"collection was not already present... added and stored embeddings")
        print(f"collection")
        reranker_model = DocumentReranker()

        res = pipeline(collection, reranker_model, query)
        print(f"response:{res}")
        df.at[index, 'response'] = res
        result = judge_llm(generated_answer=res, ground_truth=row['answers'])
        df.at[index, 'results'] = result
        time_elapsed = time() - start_time
        print(f"time elapsed:{time_elapsed}")
    
    total_time = time() - start_time
    print(f"total time taken for evaluation: {total_time}")
    df.to_csv("CUAD_evaluated.csv")


if __name__ == "__main__":
    
    main()
    