import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import sys
import math
import time
import json
import torch
import torch.nn as nn
from torch.nn import functional as F

# IMMUTABLE GROUND-TRUTH EVALUATION ENGINE - 100% SELF-CONTAINED
# ZERO IMPORTS FROM MUTABLE train.py!

SEED = 42
torch.manual_seed(SEED)

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
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
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class ImmutableLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

def get_immutable_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def evaluate_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    val_path = os.path.join(data_dir, "val.pt")
    model_path = os.path.join(current_dir, "model.pt")

    if not os.path.exists(val_path):
        print(json.dumps({"error": "Validation dataset missing", "status": "FAILED"}))
        sys.exit(1)

    if not os.path.exists(model_path):
        print(json.dumps({"error": "Trained model checkpoint missing", "status": "FAILED"}))
        sys.exit(1)

    val_dict = torch.load(val_path, weights_only=False)
    val_data = val_dict["data"]
    vocab_size = val_dict["vocab_size"]

    checkpoint = torch.load(model_path, weights_only=False)
    
    # Extract architecture metadata from checkpoint
    n_embd = checkpoint.get("n_embd", 64)
    n_head = checkpoint.get("n_head", 4)
    n_layer = checkpoint.get("n_layer", 2)
    block_size = checkpoint.get("block_size", 32)
    dropout = checkpoint.get("dropout", 0.1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ImmutableLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_batches = 30
    batch_size = 16
    losses = []

    start_time = time.time()
    with torch.no_grad():
        for _ in range(eval_batches):
            xb, yb = get_immutable_batch(val_data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())

    eval_time = time.time() - start_time
    avg_loss = sum(losses) / len(losses)
    val_bpb = avg_loss / math.log(2.0)

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    results = {
        "status": "VALID",
        "val_loss": round(avg_loss, 4),
        "val_bpb": round(val_bpb, 4),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "eval_time_sec": round(eval_time, 2),
        "trained_steps": checkpoint.get("step_count", 0)
    }

    print(json.dumps(results))

if __name__ == "__main__":
    evaluate_model()
