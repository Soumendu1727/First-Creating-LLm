import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, embed_dim, head_size, block_size):
        super().__init__()                                                        
        self.key = nn.Linear(embed_dim, head_size, bias=False)    # Key (K) → what do I contain?
        self.query = nn.Linear(embed_dim, head_size, bias=False)  # Query (Q) → what am I looking for?
        self.value = nn.Linear(embed_dim, head_size, bias=False)  # Value (V) → what information do I give?
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):  # Convert tokens → embeddings, Pass through blocks, Get logits, Calculate cross-entropy loss
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # Compute attention scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # [self.tril] This is required for GPT because GPT is autoregressive.
        wei = F.softmax(wei, dim=-1) # Turns attention scores into probabilities.

        v = self.value(x)
        out = wei @ v
        return out


    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(embed_dim, head_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

    

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(embed_dim, num_heads, block_size)
        self.ff = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x



class GPT(nn.Module):
    def __init__(self, vocab_size, block_size,
                 embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, block_size)
              for _ in range(num_layers)]
        )

        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)

        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        return logits, loss
 
