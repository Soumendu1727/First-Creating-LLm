import pickle
import pandas as pd
from tokenizer import CharTokenizer

# Load the SAME dataset
df = pd.read_csv("train.csv")

# ⚠️ IMPORTANT:
# This must match EXACTLY what you did in train.py
text = "".join(df.astype(str).values.flatten())

# Build tokenizer
tokenizer = CharTokenizer(text)

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer recreated successfully.")
print("Vocab size:", tokenizer.vocab_size)
