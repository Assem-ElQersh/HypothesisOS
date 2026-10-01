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
    # Add local directory instead of mounting as per current Modal best practices
    .add_local_dir("autoresearch", remote_path="/app/autoresearch")
)

@app.function(
    image=autoresearch_image,
    gpu="H100",          # Provision exactly 1 H100
    timeout=600,         # 10 minute absolute timeout (train loop is 5 mins)
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
            timeout=330,  # 5.5 min hard timeout inside the container
            check=False
        )
        log_content = result.stdout + "\n" + result.stderr
        
        if result.returncode == 0:
            status = "VALID"
        else:
            if "OutOfMemoryError" in log_content or "CUDA out of memory" in log_content:
                status = "OOM"
            else:
                status = "FAILED"
                
    except subprocess.TimeoutExpired as e:
        # The internal subprocess timed out
        log_content = (e.stdout.decode() if e.stdout else "") + "\n" + (e.stderr.decode() if e.stderr else "") + "\nTIMEOUT EXPIRED"
        status = "TIMEOUT"
    except Exception as e:
        log_content = str(e)
        status = "FAILED"
        
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
