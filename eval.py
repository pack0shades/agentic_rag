import re
import json
import openai
import pandas as pd
import os
from args import get_args
from main import *
from time import time
from multiprocessing import Pool, cpu_count
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    filename = filename + ".PDF"
    for dirpath, _, files in os.walk(data_dir):
        for file in files:
            if file == filename:
                final_path = os.path.join(dirpath, file)
                return final_path
    filename = filename.replace(".PDF", ".pdf")
    for dirpath, _, files in os.walk(data_dir):
        for file in files:
            if file == filename:
                final_path = os.path.join(dirpath, file)
                #print(f"File {filename} found at ::::::::{final_path}")
                return final_path
    # If the file is not found, print and return None
    print(f"File {filename} not found in {data_dir}")
    return None


def get_collection_name(file_name, json_file='collection_names.json'):
    # Remove the file extension
    collection_name = re.sub(r'\.pdf$', '', file_name, flags=re.IGNORECASE)
    # Replace invalid characters with underscores
    collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', collection_name)
    # Ensure it doesn't start or end with an invalid character
    collection_name = collection_name.strip('_-')
    # Limit to 63 characters
    if len(collection_name) > 63:
        collection_name = collection_name[:61]
        collection_name = collection_name + "A"
    
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



def process_one_batch(batch):
    results = []
    for index, row in batch.iterrows():
        query = str(row['question'])
        #print(f"{type(query)}")
        print(f"question:{query}")
        filename = row['id']
        data_dir = '/Users/bappa123/Downloads/CUAD_v1/'
        pdf_loc = find_pdf(filename=filename, data_dir=data_dir)
        # print(f"pdf_loc:{pdf_loc}")
        if pdf_loc:
            print(f" ")
        else:
            print("PDF not found.")
        collection_name = get_collection_name(filename)
        #print(f"collection name:{collection_name}")
        collection, collection_list = get_collection(collection_name)
        if collection_name not in collection_list:
            #print(f"this is collection list:{collection_list}")
            #print(f"collection was not already present... adding and storing embeddings")
            estart = time()
            embed_and_store_chunks(pdf_path=pdf_loc, doc_id=collection_name, collection=collection)
            etime = time() - estart
            #print(f"Time taken for embedding and storing: {etime}")
            #print(f"added and stored embeddings")
        '''else:
            print(" ") # print(f"collection is already present.....badhiyaa")'''
        if args.use_reranker == False:
            reranker_model = None
        elif args.reranker_model == "JinaReranker":
            reranker_model = JinaReranker()
        elif args.reranker_model == "BAAIReranker":
            reranker_model = BAAIReranker()

        print(reranker_model)
        res = pipeline(collection, reranker_model, query)
        print(f"response:{res}")
        row['response'] = res
        result = judge_llm(generated_answer=res, ground_truth=row['answers'])
        row['results'] = result
        context_retir= get_context(collection=collection,reranker=reranker_model,query=query)
        row['context_retrived']=context_retir
        results.append(row)
    
    return pd.DataFrame(results)


def main():
    start_que = args.qfrom
    end_que = args.qto
    #print(f"start_que:{start_que} end_que:{end_que}")
    logging.info("Loading data...")
    df = pd.read_csv("cuad_qas_with_responces.csv")
    df = df[start_que:end_que]
    #print(f"loaded...................")
    logging.info("Data loaded successfully.")

    num_cores = cpu_count()//2
    logging.info(f"Number of cores being used: {num_cores}")
    batch_size = len(df) // num_cores
    batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    logging.info(f"Total number of batches: {len(batches)}")

    start_time = time()
    try:
        #print(f"line:::::::::150")
        with Pool(num_cores) as pool:
            logging.info("Starting the pool processing...")
            results = pool.map(process_one_batch, batches)
            #print(f"line ::::::::::154")
            logging.info("Pool processing completed.")
    except KeyboardInterrupt:
        logging.warning("Processing interrupted by user.")
        pool.terminate()
        pool.join()
        return
    except Exception as e:
        logging.error(f"An error occurred during multiprocessing: {e}")
        return
    #print(f"line ::::::::::164")
    final_df = pd.concat(results, ignore_index=True)
    logging.info("Results combined into a single DataFrame.")
    #print(f"line ::::::::::167")
    final_df.to_csv("cuad_q1to13.csv", index=False)
    logging.info("Results saved to CSV.")
    #print(f"doneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    total_time = time() - start_time
    # print(f"Total time taken for evaluation: {total_time}")
    logging.info(f"Total time taken for evaluation: {total_time}")

if __name__ == "__main__":
    main()
