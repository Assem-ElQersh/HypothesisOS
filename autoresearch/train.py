import os
import sys
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters & Model Configuration ---
# LLM Proposal Engine can modify these hyperparameters or neural architecture components below.
batch_size = 16
block_size = 32
max_iters = 300
eval_interval = 50
learning_rate = 0.003
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.1

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Simple linear layer followed by non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    train_path = os.path.join(data_dir, "train.pt")

    if not os.path.exists(train_path):
        from prepare import prepare_dataset
        prepare_dataset(data_dir)

    dataset_dict = torch.load(train_path, weights_only=False)
    train_data = dataset_dict["data"]
    vocab_size = dataset_dict["vocab_size"]

    model = LanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"[Train] Starting PyTorch model training on {device} (iters: {max_iters}, lr: {learning_rate})...")
    start_time = time.time()

    for iter in range(max_iters):
        xb, yb = get_batch(train_data, block_size, batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % eval_interval == 0 or iter == max_iters - 1:
            print(f"[Train] iter {iter:4d}: loss {loss.item():.4f}")

    training_time = time.time() - start_time
    save_path = os.path.join(current_dir, "model.pt")
    torch.save({"model_state_dict": model.state_dict(), "vocab_size": vocab_size, "n_embd": n_embd, "block_size": block_size}, save_path)
    print(f"[Train] Model training complete in {training_time:.2f}s. Saved to {save_path}.")

if __name__ == "__main__":
    train_model()
