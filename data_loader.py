import pandas as pd
import torch

def load_csv_text(file):
    df = pd.read_csv(file)
    df = df.fillna("")
    df["text_clean"] = df.astype(str).agg(" ".join, axis=1)
    return "\n".join(df["text_clean"])

def get_batch(data, block_size=128, batch_size=32):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
