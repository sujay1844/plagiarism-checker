import pandas as pd
df = pd.read_csv("./train_snli.txt", sep="\t", header=None, names=['sentence1', 'sentence2', 'label'])
df.to_csv('./train_snli.csv', index=False)
