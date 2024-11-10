import re
import hashlib
import openai
from openai import OpenAI
import pandas as pd
from config import EVAL_PROMPT_SYS, EVAL_PROMPT_USR, MODEL
import os
from args import get_args
from main import generate_response_from_context
from main import (
    generate_response_from_context,
    get_context,
    generate_response_from_multi_agent
)
from agent import context_to_agent
from chromadb_client import get_collection
from chroma_db import (
    embed_and_store_chunks
)
from reranker import JinaReranker, BAAIReranker
import time
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import logging
import pandas as pd
from dotenv import load_dotenv
from chromadb_client import get_collection, retrieve_documents
import chromadb
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # For FastAPI / Starlette
logging.getLogger("werkzeug").setLevel(logging.WARNING)        # For Flask
logging.getLogger("http.client").setLevel(logging.WARNING)     # For Python's http.client
logging.getLogger("urllib3").setLevel(logging.WARNING)         # For urllib3, used in requests

args = get_args()

openaiclient = OpenAI()

def generate_response(
    client: openai.OpenAI, system_prompt: str, user_prompt: str, model: str
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


def judge_eval(ground_truth: str, predict_answer: str) -> int:

    eval_response = generate_response(
        client=openaiclient,
        system_prompt=EVAL_PROMPT_SYS,
        user_prompt=EVAL_PROMPT_USR.format(ground_truth, predict_answer),
        model=MODEL,
    )

    return int(eval_response) if eval_response in {"1", "0"} else 0


def eval_function(dataset: pd.DataFrame) -> pd.DataFrame:

    results_df = []
    for _, row in dataset.iterrows():
        row["results"] = judge_eval(row["answers"], row["response"])
        results_df.append(row)

    pd.DataFrame(results_df)
    # percent_of_ones = (results_df["results"].sum() / len(results_df)) * 100
    return results_df


def eval_pipeline_(
    dataset: pd.DataFrame, collection: chromadb.Collection, client: chromadb.Client
) -> pd.DataFrame:
    
    if args.use_reranker == False:
        reranker_model = None
    elif args.reranker_model == "JinaReranker":
        reranker_model = JinaReranker()
    elif args.reranker_model == "BAAIReranker":
        reranker_model = BAAIReranker()
    else:
        print ("reranker model not found")
        exit(0)
    
    results = []
    for index, row in dataset.iterrows():

        # print ('0.00000000000000000000000000000000000000000000000000000000000000')

        query = str(row["question"])
        context = get_context(
            collection=collection, reranker=None, query=query
        )

        res = ""

        if args.pipeline == "multi_agent":
            fin_context = context_to_agent(context)
            res = generate_response_from_context(query, fin_context)

        elif args.pipeline == "router":
            res = generate_response_from_multi_agent(query, context)

        elif args.pipeline == "naive":
            res = generate_response_from_context(query, context)
            
        else:
            logging.error("use --pipeline argument to specify the pipeline")
            exit(0)
            
        row["context_retrived"] = context
        row["response"] = generate_response_from_context(query, context)

        results.append(row)


    results = pd.DataFrame(results)
    # print ('main pipeline result -------------- {} '.format(results.shape))
    
    return results


class EvaluationPipeline(object):
    def __init__(self, main_pipeline, eval_function, dataset: pd.DataFrame) -> None:
        self.pipeline = main_pipeline
        self.eval_function = eval_function
        self.dataset = dataset

    def run_eval_(self) -> None:
        if not self.eval_function:
            logging.error("Eval function not provided")
            return None
        else:
            logging.info(f"Eval Metrics Given {self.eval_function}")

        filename = self.dataset.iloc[0][0]
        # print (f"file name - {filename}")
        data_dir = os.getenv("DATA_DIR")
        pdf_loc = find_pdf(filename=filename, data_dir=data_dir)

        if pdf_loc is None:
            logging.warning(f"File {filename} not found.")
            return None  # Return None if file is not found

        collection_name = get_collection_name(filename)
        collection, client = get_collection(collection_name)
        pdfpath = find_pdf(data_dir=data_dir, filename=filename)
        embed_and_store_chunks(pdf_path=pdfpath, doc_id=collection_name, collection=collection)

        results = self.pipeline(self.dataset, collection, client)

        if results is None:
            logging.error(f"Pipeline did not return valid results for {filename}")
            return None  # Return None if results are not generated

        results = eval_function(results)
        # print ('results formed')
        ## delete collection
        client.delete_collection(collection_name)
        return results, collection_name


def find_pdf(data_dir: str, filename: str
)-> str:
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
                # print(f"File {filename} found at ::::::::{final_path}")
                return final_path
    # If the file is not found, print and return None
    print(f"File {filename} not found in {data_dir}")
    return None


def get_collection_name(file_name: str) -> str:
    # Remove the .pdf extension if present (case insensitive)
    collection_name = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
    collection_name = re.sub(r"\.PDF$", "", file_name, flags=re.IGNORECASE)
    collection_name = re.sub(r"[^a-zA-Z0-9._-]", "_", collection_name)
    collection_name = re.sub(r"\.\.+", "_", collection_name)
    collection_name = collection_name.strip("_-")

    if len(collection_name) > 50:
        collection_name = collection_name[:50]

    unique_suffix = hashlib.md5(file_name.encode()).hexdigest()[:10]  
    unique_suffix = unique_suffix[:12]  + "a"
    collection_name = f"{collection_name}_{unique_suffix}"

    return collection_name


def process_one_batch(batch: pd.DataFrame = None) -> pd.DataFrame:
    # print('in process_one_batch()')
    evalpipeline = EvaluationPipeline(main_pipeline=eval_pipeline_, eval_function=eval_function, dataset=batch)
    # eval_function(evalpipeline(dataset=batch))
    # print ('run_eval')
    result = evalpipeline.run_eval_()

    if result is None:
        logging.error("Evaluation pipeline returned None.")
        return pd.DataFrame()  # Return an empty DataFrame or handle as needed

    results, collection_name = result
    logging.info(f"Results for {collection_name} saved to CSV.")
    results = pd.DataFrame(results)
    results.to_csv('results/' + collection_name + ".csv", index=False)
    logging.info(f"sleeeping.......................")
    time.sleep(10)
    return results



def main():
    logging.info("Loading data...")
    df = pd.read_csv("./cuad_qas_with_responces.csv")
    logging.info("Data loaded successfully.")

    num_cores = 6
    logging.info(f"Number of cores being used: {num_cores}")

    # Create actual DataFrame batches
    batches = [group for _, group in df.groupby("id")][args.dfrom:args.dto]
    start_time = time.time()
    print ('num of batches - ',len(batches))

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_one_batch, batches), total=len(batches)))

    final_df = pd.concat(results, ignore_index=True)
    logging.info("Results combined into a single DataFrame.")
    # fin_acc = (final_df["results"].sum() / len(final_df)) * 100
    # logging.info(f"final accuracy : {fin_acc}")

    final_df.to_csv("results/results.csv", index=False, mode='a')
    logging.info("Results saved to CSV.")

    total_time = time.time() - start_time

    logging.info(f"Total time taken for evaluation: {total_time}")


if __name__ == "__main__":
    main()
