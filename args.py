import argparse

def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("--use_reranker", type=bool, default=True, help="Use reranker or not")
    arg.add_argument("--retrieved_docs", type=int, default=5, help="Number of retrieved documents")
    return  arg.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)