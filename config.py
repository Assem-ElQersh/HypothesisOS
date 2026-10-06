import os

class Config:
    # OpenRouter API settings
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Model selections (decoupled from orchestrator logic)
    PROPOSER_MODEL = os.getenv("PROPOSER_MODEL", "google/gemini-2.5-pro")
    CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "google/gemini-2.5-pro")
    COUNCIL_MODELS = [
        "google/gemini-2.5-pro",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.3-70b-instruct"
    ]
    
    # Execution & Hardware Settings
    TRAIN_TIME_BUDGET_SEC = 60         # Default time budget per training trial (in seconds)
    EXECUTION_TIMEOUT_SEC = 120        # Hard timeout for training process
    DEFAULT_MAX_EXPERIMENTS = 5        # Number of loop iterations for autonomous run
    
    # Project paths
    WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
    AUTORESEARCH_DIR = os.path.join(WORKSPACE_DIR, "autoresearch")
    TRAIN_SCRIPT = os.path.join(AUTORESEARCH_DIR, "train.py")
    EVALUATE_SCRIPT = os.path.join(AUTORESEARCH_DIR, "evaluate.py")
    PREPARE_SCRIPT = os.path.join(AUTORESEARCH_DIR, "prepare.py")
    
    # State & Memory paths
    MEMORY_DIR = os.path.join(WORKSPACE_DIR, "memory")
    RESEARCH_TREE_PATH = os.path.join(MEMORY_DIR, "research_tree.json")
    RESEARCH_PLAN_PATH = os.path.join(WORKSPACE_DIR, "research_plan.md")
    RESEARCH_LEDGER_PATH = os.path.join(WORKSPACE_DIR, "research_ledger.tsv")
    
    # Prompt Caching headers for OpenRouter
    PROMPT_CACHE_HEADERS = {
        "HTTP-Referer": "https://github.com/Assem-ElQersh/HypothesisOS",
        "X-Title": "HypothesisOS Autonomous Researcher"
    }

config = Config()
