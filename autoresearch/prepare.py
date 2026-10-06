import os
import torch
import math

def prepare_dataset(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    print("[Prepare] Generating dataset...")
    # Synthetic text sequence dataset or TinyShakespeare text
    text = (
        "First Citizen:\n"
        "Before we proceed any further, hear me speak.\n\n"
        "All:\n"
        "Speak, speak.\n\n"
        "First Citizen:\n"
        "You are all resolved rather to die than to famish?\n\n"
        "All:\n"
        "Resolved. resolved.\n\n"
        "First Citizen:\n"
        "First, you know Caius Marcius is chief enemy to the people.\n\n"
        "All:\n"
        "We know't, we know't.\n\n"
        "First Citizen:\n"
        "Let us kill him, and we'll have corn at our own price.\n"
        "Is't a verdict?\n\n"
        "All:\n"
        "No more talking on't; let it be done: away, away!\n"
    ) * 100

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)

    # Train / Val split (90% / 10%)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    torch.save({"data": train_data, "vocab_size": vocab_size, "char_to_ix": char_to_ix, "ix_to_char": ix_to_char}, train_path)
    torch.save({"data": val_data, "vocab_size": vocab_size, "char_to_ix": char_to_ix, "ix_to_char": ix_to_char}, val_path)
    print(f"[Prepare] Dataset prepared: Train length {len(train_data)}, Val length {len(val_data)}, Vocab size {vocab_size}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prepare_dataset(os.path.join(current_dir, "data"))
