import torch
import torch.optim as optim

from tokenizer import CharTokenizer
from data_loader import load_csv_text, get_batch
from model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load datasets
train_text = load_csv_text("train.csv")
val_text = load_csv_text("validation.csv")
test_text = load_csv_text("test.csv")

# Tokenizer (only on train data)
tokenizer = CharTokenizer(train_text)

train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
test_data = torch.tensor(tokenizer.encode(test_text), dtype=torch.long)

block_size = 128

model = GPT(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)  # Good for Transformers, Without this model never learns

steps = 5000

for step in range(steps):

    xb, yb = get_batch(train_data, block_size)
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb) # Passes data through GPT, Computes cross entropy loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {step}")
    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch(val_data, block_size)
            xv, yv = xv.to(device), yv.to(device)
            _, val_loss = model(xv, yv)
        model.train()

        print(f"Step {step}")
        print("Train Loss:", loss.item())
        print("Val Loss:", val_loss.item())
        print("-" * 30)

torch.save(model.state_dict(), "model.pt")
print("Training Complete")

# Final test evaluation
model.eval()
with torch.no_grad():
    xt, yt = get_batch(test_data, block_size)
    xt, yt = xt.to(device), yt.to(device)
    _, test_loss = model(xt, yt)

print("Final Test Loss:", test_loss.item())
