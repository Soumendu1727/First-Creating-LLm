import torch
import pickle
from model import GPT

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = tokenizer.vocab_size
block_size = 128  # ⚠ must be SAME as training

# Load model
model = GPT(vocab_size, block_size)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()


def generate(model, start_text, max_new_tokens=50, temperature=0.3):
    input_ids = torch.tensor(
        [tokenizer.encode(start_text)],
        dtype=torch.long
    ).to(device)

    for _ in range(max_new_tokens):

        # Crop to block size
        idx_cond = input_ids[:, -block_size:]

        # Forward pass
        logits, _ = model(idx_cond)

        # Take last token
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids[0].tolist())


# Example usage
output = generate(model, "DUKE VINCENTIO\n", max_new_tokens=50, temperature=0.3)
print(f'Answer: {output}')
