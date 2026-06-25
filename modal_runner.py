import modal
import os
import subprocess

app = modal.App("hypothesis-os-executor")

# 1. Build a persistent Image with all dependencies and the downloaded dataset embedded
autoresearch_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", 
        "numpy", 
        "pandas", 
        "tiktoken", 
        "rustbpe", 
        "hf-xet",
        "huggingface-hub",
        "pygments",
        "matplotlib",
        "networkx",
        "kiwisolver",
        "pillow",
        "sympy"
    )
    .add_local_file("autoresearch/prepare.py", "/app/prepare.py")
    .run_commands(
        "cd /app && python prepare.py"
    )
)

# 2. Mount the local autoresearch directory so the latest train.py is synced instantly
autoresearch_mount = modal.Mount.from_local_dir(
    "autoresearch", 
    remote_path="/app/autoresearch",
    condition=lambda p: not p.endswith(".venv") and "__pycache__" not in p
)

@app.function(
    image=autoresearch_image,
    gpu="H100",          # Provision exactly 1 H100
    timeout=600,         # 10 minute absolute timeout (train loop is 5 mins)
    mounts=[autoresearch_mount]
)
def run_training_experiment():
    """Runs train.py on the Modal H100 and returns the log."""
    os.chdir("/app/autoresearch")
    
    print("Starting H100 Training Run (5 minute budget)...")
    try:
        # We run the train script and capture the log
        result = subprocess.run(
            ["python", "train.py"], 
            capture_output=True, 
            text=True,
            check=False
        )
        log_content = result.stdout + "\n" + result.stderr
        status = "SUCCESS" if result.returncode == 0 else "CRASH"
        
    except Exception as e:
        log_content = str(e)
        status = "CRASH"
        
    return status, log_content

@app.local_entrypoint()
def main():
    print("Dispatching experiment to Modal H100...")
    status, log_content = run_training_experiment.remote()
    
    print(f"Modal execution finished with status: {status}")
    
    # Save the log locally so the orchestrator can read it
    with open("autoresearch/run.log", "w", encoding="utf-8") as f:
        f.write(log_content)
        
    print("Log saved to autoresearch/run.log.")
