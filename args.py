import argparse

def get_args():
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--use_reranker", type=bool, default=True, help="Use reranker or not")
    argparse.add_argument("--retrieved_docs", type=int, default=5, help="Number of retrieved documents")
    return  argparse.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)