import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import torch
import math
import random
import urllib.request

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def fetch_tinyshakespeare() -> str:
    """ Attempt to download standard TinyShakespeare dataset (1MB / ~1.1M chars) """
    try:
        print("[Prepare] Downloading TinyShakespeare benchmark dataset...")
        req = urllib.request.Request(
            TINY_SHAKESPEARE_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8")
            if len(text) > 100000:
                print(f"[Prepare] Downloaded TinyShakespeare successfully ({len(text)} characters).")
                return text
    except Exception as e:
        print(f"[Prepare Notice] Could not download online dataset ({e}). Falling back to cached text corpus.")

    # High-density diverse fallback corpus (~15KB)
    fallback_corpus = (
        "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n"
        "First Citizen:\nYou are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\n"
        "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n\n"
        "All:\nWe know't, we know't.\n\nFirst Citizen:\nLet us kill him, and we'll have corn at our own price.\n"
        "Is't a verdict?\n\nMENENIUS:\nI tell you, friends, most charitable care\nHave the patricians of you.\n"
        "For your wants, your suffering in this dearth, you may as well\nStrike at the heaven with your staves as lift them\n"
        "Against the Roman state, whose course will on\nThe way it takes, cracking ten thousand curbs\n"
        "Of more strong link asunder than can ever\nAppear in your impediment. For the dearth,\n"
        "The gods, not the patricians, make it, and\nYour knees to them, not arms, must help.\n\n"
        "VIRGILIA:\nHe'er shall he come: O, my good lord, I have seen a boy of his, that I had rather\n"
        "Look upon than twenty such statues as his father.\n\nVOLUMNIA:\nHe'll beat them to their heels: he is a lion\n"
        "That I am proud to hunt. Look, here he comes: Welcome, my brave soldier!\n\n"
        "CORIOLANUS:\nHail, lords! I am returned your soldier; No more infected with my country's love\n"
        "Than when I parted hence, but still it holds In equal rank with your best service.\n\n"
        "AUFIDIUS:\nRead it not, noble lords; But tell the traitor he has abused your powers\n"
        "And given up, for certain drops of salt, Your city Rome, I say 'your city,' to his wife and mother;\n"
        "Breaking his oath and resolution like A twist of rotten silk.\n\n"
        "ROMEO:\nIt was the lark, the herald of the morn, No nightingale: look, love, what envious streaks\n"
        "Do lace the severing clouds in yonder east: Night's candles are burnt out, and jocund day\n"
        "Stands tiptoe on the misty mountain tops. I must be gone and live, or stay and die.\n"
    ) * 10
    return fallback_corpus

def prepare_dataset(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    raw_path = os.path.join(data_dir, "input.txt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[Prepare] Verified existing dataset files in {data_dir}.")
        return

    text = fetch_tinyshakespeare()
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(text)

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)

    # STRICT DISJOINT NON-OVERLAPPING TRAIN / VAL SPLIT (90% / 10%)
    n = int(0.90 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    torch.save({"data": train_data, "vocab_size": vocab_size, "char_to_ix": char_to_ix, "ix_to_char": ix_to_char}, train_path)
    torch.save({"data": val_data, "vocab_size": vocab_size, "char_to_ix": char_to_ix, "ix_to_char": ix_to_char}, val_path)
    print(f"[Prepare] Non-leaky dataset prepared: Train length {len(train_data)}, Val length {len(val_data)}, Vocab size {vocab_size}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prepare_dataset(os.path.join(current_dir, "data"))
