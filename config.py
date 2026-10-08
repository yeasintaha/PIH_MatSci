import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
FIREWORKS_API_KEY = os.getenv("FIREWORKS", "")
DEFAULT_MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_DATASET_PATH = PROJECT_ROOT.parent / "material" / "matscibench.json"

CLEAN_DATASET_PATH = DATA_DIR / "matscibench_clean.json"
PIH_DATASET_PATH = DATA_DIR / "pih_dataset.json"

BASELINE_RESULTS_PATH = RESULTS_DIR / "baseline_results.json"
PIH_RESULTS_PATH = RESULTS_DIR / "pih_results.json"
JUDGE_RESULTS_PATH = RESULTS_DIR / "judge_results.json"
MITIGATION_RESULTS_PATH = RESULTS_DIR / "mitigation_results.json"
HUMAN_VALIDATION_SAMPLE_PATH = RESULTS_DIR / "human_validation_sample.json"

for d in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


ATTACKER_MODEL = {
    "provider": "openai",
    "model_id": "gpt-4o",
    "display_name": "GPT-4o (Attacker)",
}

JUDGE_MODEL = {
    "provider": "anthropic",
    "model_id": "claude-sonnet-4-20250514",
    "display_name": "Claude Sonnet 4 (Judge)",
}

TARGET_MODELS = [
    # {
    #     "provider": "fireworks",
    #     "model_id": "accounts/fireworks/models/gpt-oss-120b",
    #     "display_name": "GPT-OSS-120B",
    # },
    {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-r1",
        "display_name": "DeepSeek-R1",
    },
    # {
    #     "provider": "openrouter",
    #     "model_id": "deepseek/deepseek-chat",
    #     "display_name": "DeepSeek-V3",
    # },
]

TARGET_TEMPERATURE = 0.0

JUDGE_TEMPERATURE = 0.0

ATTACKER_TEMPERATURE = 0.7

TARGET_MAX_TOKENS = 16384
JUDGE_MAX_TOKENS = 1024
ATTACKER_MAX_TOKENS = 512
ANSWER_TOLERANCE = 0.05

RATE_LIMITS = {
    "openai": 60,
    "anthropic": 40,
    "openrouter": 60,
    "fireworks": 120,
    "google": 60,
}

EXPERIMENT_NAME = "pih_matscibench_v1"
RANDOM_SEED = 42
