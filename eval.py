import openai
import pandas as pd
import os


def evaluate_answer(ground_truth, generated_answer):
    prompt = f"""
    Ground Truth: "{ground_truth}"
    Generated Answer: "{generated_answer}"
    Evaluate the similarity of two given text snippets. Input: Ground Truth Text, Predicted Answer Text. Output: 1 if the Predicted Answer Text has the same context as the Ground Truth Text, 0 otherwise.
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

if __name__ == "__main__":
    df = pd.read_csv("your_file.csv")

    df["bool_val"] = df.apply(lambda row: evaluate_answer(row["ground_truth"], row["generated_answer"]), axis=1)
    df.to_csv("eval.csv", index=False)
    correct_sum = df["bool_val"].sum()
    num_columns = df.shape[1]
    print(100*(correct_sum/num_columns))


