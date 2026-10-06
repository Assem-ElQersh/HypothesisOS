import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import torch
import math
import random

def prepare_dataset(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    print("[Prepare] Generating non-leaky benchmark dataset...")
    
    # Real diverse, non-repeating character-level text corpus (TinyShakespeare excerpts)
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
        "MENENIUS:\n"
        "I tell you, friends, most charitable care\n"
        "Have the patricians of you. For your wants,\n"
        "Your suffering in this dearth, you may as well\n"
        "Strike at the heaven with your staves as lift them\n"
        "Against the Roman state, whose course will on\n"
        "The way it takes, cracking ten thousand curbs\n"
        "Of more strong link asunder than can ever\n"
        "Appear in your impediment. For the dearth,\n"
        "The gods, not the patricians, make it, and\n"
        "Your knees to them, not arms, must help.\n\n"
        "VIRGILIA:\n"
        "He'er shall he come: O, my good lord,\n"
        "I have seen a boy of his, that I had rather\n"
        "Look upon than twenty such statues as his father.\n\n"
        "VOLUMNIA:\n"
        "He'll beat them to their heels: he is a lion\n"
        "That I am proud to hunt. Look, here he comes:\n"
        "Welcome, my brave soldier!\n\n"
        "CORIOLANUS:\n"
        "Hail, lords! I am returned your soldier;\n"
        "No more infected with my country's love\n"
        "Than when I parted hence, but still it holds\n"
        "In equal rank with your best service.\n\n"
        "AUFIDIUS:\n"
        "Read it not, noble lords;\n"
        "But tell the traitor he has abused your powers\n"
        "And given up, for certain drops of salt,\n"
        "Your city Rome, I say 'your city,' to his wife and mother;\n"
        "Breaking his oath and resolution like\n"
        "A twist of rotten silk.\n\n"
        "SCENE II. A street near the Forum.\n"
        "Enter ROMEO and JULIET above, at the window.\n\n"
        "JULIET:\n"
        "Wilt thou be gone? it is not yet near day:\n"
        "It was the nightingale, and not the lark,\n"
        "That pierced the fearful hollow of thine ear;\n"
        "Nightly she sings on yon pomegranate-tree:\n"
        "Believe me, love, it was the nightingale.\n\n"
        "ROMEO:\n"
        "It was the lark, the herald of the morn,\n"
        "No nightingale: look, love, what envious streaks\n"
        "Do lace the severing clouds in yonder east:\n"
        "Night's candles are burnt out, and jocund day\n"
        "Stands tiptoe on the misty mountain tops.\n"
        "I must be gone and live, or stay and die.\n"
    )

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)

    # STRICT DISJOINT NON-OVERLAPPING TRAIN / VAL SPLIT (85% / 15%)
    n = int(0.85 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    torch.save({"data": train_data, "vocab_size": vocab_size, "char_to_ix": char_to_ix, "ix_to_char": ix_to_char}, train_path)
    torch.save({"data": val_data, "vocab_size": vocab_size, "char_to_ix": char_to_ix, "ix_to_char": ix_to_char}, val_path)
    print(f"[Prepare] Non-leaky dataset prepared: Train length {len(train_data)}, Val length {len(val_data)}, Vocab size {vocab_size}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prepare_dataset(os.path.join(current_dir, "data"))
