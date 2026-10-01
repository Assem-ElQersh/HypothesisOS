import time
import random

def main():
    print("Starting training...")
    time.sleep(1)
    # The patch will change this code
    val_bpb = 3.42 + random.uniform(-0.05, 0.05)
    peak_vram_mb = 18000
    print(f"val_bpb: {val_bpb:.4f}")
    print(f"peak_vram_mb: {peak_vram_mb}")

if __name__ == "__main__":
    main()
