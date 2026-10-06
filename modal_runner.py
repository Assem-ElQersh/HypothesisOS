import modal
import os
import subprocess

app = modal.App("hypothesis-os-executor")

# Persistent Image with PyTorch & scientific dependencies
autoresearch_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        "tiktoken",
        "matplotlib"
    )
    .add_local_dir("autoresearch", remote_path="/app/autoresearch")
)

@app.function(
    image=autoresearch_image,
    gpu="any",            # Flexible GPU allocation
    timeout=300,          # 5 minute execution timeout
)
def run_training_experiment():
    """Runs PyTorch training and immutable evaluation in Modal container."""
    os.chdir("/app/autoresearch")

    print("[Modal Container] Running dataset preparation script...")
    subprocess.run(["python", "prepare.py"], check=False)

    print("[Modal Container] Running PyTorch train script...")
    train_res = subprocess.run(["python", "train.py"], capture_output=True, text=True, check=False)
    train_log = train_res.stdout + "\n" + train_res.stderr

    if train_res.returncode != 0:
        return "CRASH", train_log

    print("[Modal Container] Running Immutable Evaluator...")
    eval_res = subprocess.run(["python", "evaluate.py"], capture_output=True, text=True, check=False)
    eval_log = eval_res.stdout

    full_log = train_log + "\n--- IMMUTABLE EVALUATION OUTPUT ---\n" + eval_log
    status = "VALID" if eval_res.returncode == 0 else "FAILED"

    return status, full_log

@app.local_entrypoint()
def main():
    print("[Modal Dispatcher] Offloading PyTorch training run to Modal...")
    status, log_content = run_training_experiment.remote()

    print(f"[Modal Dispatcher] Container execution finished with status: {status}")
    with open("autoresearch/run.log", "w", encoding="utf-8") as f:
        f.write(log_content)

    print("Log saved to autoresearch/run.log.")
