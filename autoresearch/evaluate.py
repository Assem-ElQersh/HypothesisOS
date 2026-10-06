import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
from torch.nn import functional as F

# Ground-truth evaluation engine - IMMUTABLE BY AGENT PROPOSALS

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

    # Import LanguageModel class dynamically from train.py
    sys.path.insert(0, current_dir)
    from train import LanguageModel, block_size, device, batch_size, get_batch

    checkpoint = torch.load(model_path, weights_only=False)
    model = LanguageModel(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_batches = 20
    losses = []

    start_time = time.time()
    with torch.no_grad():
        for _ in range(eval_batches):
            xb, yb = get_batch(val_data, block_size, batch_size)
            _, loss = model(xb, yb)
            losses.append(loss.item())

    eval_time = time.time() - start_time
    avg_loss = sum(losses) / len(losses)
    # val_bpb = val_loss / ln(2)
    val_bpb = avg_loss / math.log(2.0)

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    results = {
        "status": "VALID",
        "val_loss": round(avg_loss, 4),
        "val_bpb": round(val_bpb, 4),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "eval_time_sec": round(eval_time, 2)
    }

    print(json.dumps(results))

if __name__ == "__main__":
    evaluate_model()
