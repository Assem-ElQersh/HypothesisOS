import os

class Config:
    # Try loading .env file if present
    _env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_file):
        with open(_env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                if "=" in _line and not _line.strip().startswith("#"):
                    _k, _v = _line.strip().split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

    # OpenRouter API settings
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Model selections (decoupled from orchestrator logic - using OpenRouter free models by default)
    PROPOSER_MODEL = os.getenv("PROPOSER_MODEL", "google/gemini-2.5-flash:free")
    CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    COUNCIL_MODELS = [
        os.getenv("MODEL_1", "google/gemini-2.5-flash:free"),
        os.getenv("MODEL_2", "meta-llama/llama-3.3-70b-instruct:free"),
        os.getenv("MODEL_3", "deepseek/deepseek-r1:free")
    ]
    
    # Execution & Hardware Settings
    EXECUTION_BACKEND = os.getenv("EXECUTION_BACKEND", "local") # "local" or "modal"
    TRAIN_TIME_BUDGET_SEC = int(os.getenv("TRAIN_TIME_BUDGET_SEC", "30")) # Fixed wall-clock compute budget per trial
    EXECUTION_TIMEOUT_SEC = int(os.getenv("EXECUTION_TIMEOUT_SEC", "60")) # Hard process timeout limit
    DEFAULT_MAX_EXPERIMENTS = int(os.getenv("DEFAULT_MAX_EXPERIMENTS", "3"))
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42")) # Global seed for determinism
    
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
