import openai
import pandas as pd
import os
from args import get_args
from main import *
from time import time

args = get_args()

def judge_llm(ground_truth, generated_answer):
    prompt1 = f"""
    
    Evaluate the similarity of two given text snippets. Input: Ground Truth Text, Predicted Answer Text.
    Output: 1 if the Predicted Answer Text has the same context as the Ground Truth Text, 0 otherwise 
    '''do not answer anything except 0 or 1'''.
    """
    prompt2 = f"""
    Ground Truth: "{ground_truth}"
    Generated Answer: "{generated_answer}"
    
    """

    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": prompt2},
        ]
    )

    output = response.choices[0].message.content
    print(f"judge saahab ka decision: {output}")
    return int(output) if output in {"1", "0"} else 0

def find_pdf(data_dir: str, filename: str) -> str:
    filename = filename + ".PDF"
    for dirpath, _, files in os.walk(data_dir):
        for file in files:
            if file == filename:
                final_path = os.path.join(dirpath, file)
                return final_path
    
    # If the file is not found, print and return None
    print(f"File {filename} not found in {data_dir}")
    return None
            
import re
import json
import os

def get_collection_name(file_name, json_file='collection_names.json'):
    # Remove the file extension
    collection_name = re.sub(r'\.pdf$', '', file_name, flags=re.IGNORECASE)
    # Replace invalid characters with underscores
    collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', collection_name)
    # Ensure it doesn't start or end with an invalid character
    collection_name = collection_name.strip('_-')
    # Limit to 63 characters
    collection_name = collection_name[:63]
    
    # Store the original filename and new collection name in a JSON file
    if os.path.exists(json_file):
        # Load existing data
        with open(json_file, 'r') as f:
            collection_data = json.load(f)
    else:
        # Initialize new data
        collection_data = {}

    # Add or update the mapping
    collection_data[file_name] = collection_name
    
    # Save the updated data back to the JSON file
    with open(json_file, 'w') as f:
        json.dump(collection_data, f, indent=4)
    
    return collection_name




def main():
    print(f"Using Reranker: {args.use_reranker}_____number of Retrieved Docs: {args.retrieved_docs}")
    df = pd.read_csv("cuad_qas_with_responces.csv")
    start_time = time()

    for index, row in df.iterrows():
        query = str(row['question'])
        print(f"{type(query)}")
        print(f"question:{query}")
        filename = row['id']
        data_dir = '/home/pragay/interiit/CUAD_v1/'
        pdf_loc = find_pdf(filename=filename, data_dir=data_dir)
        if pdf_loc:
            print(f"PDF found at: {pdf_loc}")
        else:
            print("PDF not found.")
        collection_name = get_collection_name(filename)
        print(f"collection name:{collection_name}")
        collection, collection_present = get_collection(collection_name)
        if collection_present==False:
            embed_and_store_chunks(pdf_path=pdf_loc,doc_id=collection_name, collection=collection)
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
    