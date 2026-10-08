import json
import logging
import math
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

_jsonl_write_lock = threading.Lock()

from config import LOGS_DIR, ANSWER_TOLERANCE



def setup_logging(step_name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("pih")
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (with UTF-8 support for Windows)
    import io
    stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console = logging.StreamHandler(stream)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    log_file = LOGS_DIR / f"{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")
    return logger


# ─────────────────────────────────────────────
# JSON I/O
# ─────────────────────────────────────────────

def load_json(path: Union[str, Path]) -> Union[list, dict]:
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Union[list, dict], path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def append_jsonl(record: dict, path: Union[str, Path]) -> None:
    """Append a single record to a JSONL file (crash-safe incremental saving, thread-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _jsonl_write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path], deduplicate: bool = True) -> list[dict]:
    """Load all records from a JSONL file. Deduplicates by experiment_id by default."""
    records = []
    if not Path(path).exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                
    if deduplicate:
        # Keep only the latest entry for each experiment_id
        deduped = {}
        for r in records:
            eid = r.get("experiment_id")
            if eid is not None:
                deduped[eid] = r
            else:
                deduped[id(r)] = r
        return list(deduped.values())
        
    return records



def get_completed_ids(jsonl_path: Union[str, Path]) -> set:
    """Get the set of experiment_ids already completed successfully (for resumable runs)."""
    path = Path(jsonl_path)
    if not path.exists():
        return set()
    completed = set()
    for record in load_jsonl(path, deduplicate=True):
        eid = record.get("experiment_id")
        if not eid:
            continue
            
        is_failed = (
            bool(record.get("error")) or
            record.get("constraint_type") == "parse_failure" or
            record.get("combined_label") == "PARSE_ERROR"
        )
        
        if not is_failed:
            completed.add(eid)
            
    return completed


def timestamp() -> str:
    """Get current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────────

def progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> str:
    """Simple text progress bar."""
    frac = current / total if total > 0 else 0
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100
    return f"{prefix}|{bar}| {current}/{total} ({pct:.1f}%)"
