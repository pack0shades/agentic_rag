import os
import pandas as pd

# Define the path to your directory
directory_path = '/home/pragay/interiit/agentic_rag/results_multi'

dataframes = []

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

result = combined_df['results'].sum() / len(combined_df)
print(result)
