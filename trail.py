import pandas as pd
df = pd.read_csv("cuad_qas_with_responces.csv")
df.drop([0],axis=0, inplace=True)
df.to_csv("cuad_qas_with_responces.csv")