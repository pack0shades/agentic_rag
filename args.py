import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("--use_reranker", type=bool, default=False, help="Use reranker or not")
    arg.add_argument("--retrieved_docs", type=int, default=5, help="Number of retrieved documents")
    arg.add_argument("--collection_name", type=str, default="generate", help="Name of the collection")
    arg.add_argument("--pdf_path", type=str, default="./pdfs/nvidia.pdf", help="Path to the PDF document")
    return  arg.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
