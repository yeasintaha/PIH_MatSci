import json
import logging
import math
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

_jsonl_write_lock = threading.Lock()

from config import LOGS_DIR, ANSWER_TOLERANCE, MATSCIBENCH_TOLERANCE, TRUNCATION_FINISH_REASONS



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

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    console = logging.StreamHandler(sys.stdout)
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


# Record identity

def make_record_key(experiment_id: Optional[str], sample_idx: Optional[int] = 0) -> str:
    return f"{experiment_id}::{int(sample_idx or 0)}"


def record_key(record: dict) -> str:
    return make_record_key(record.get("experiment_id"), record.get("sample_idx"))


def base_experiment_id(record_or_id: Union[dict, str, None]) -> str:
    """
    Base problem ID of a perturbation record, e.g. 'PIH_0003_type_II' -> 'PIH_0003' and
    'PIH_IMG_0003_type_I' -> 'PIH_IMG_0003'. Uses the explicit base_experiment_id field when present.
    """
    if isinstance(record_or_id, dict):
        explicit = record_or_id.get("base_experiment_id")
        if explicit:
            return explicit
        record_or_id = record_or_id.get("experiment_id")
    return str(record_or_id or "").split("_type_")[0]


def is_truncated(record: dict) -> bool:
    """True if the generation hit the token ceiling (or a judge record marked it Truncated)."""
    return (
        record.get("finish_reason") in TRUNCATION_FINISH_REASONS
        or record.get("external_compliance") == "Truncated"
    )


# JSON I/O

def load_json(path: Union[str, Path]) -> Union[list, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Union[list, dict], path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def append_jsonl(record: dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _jsonl_write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path], deduplicate: bool = True) -> list[dict]:
    """
    Load all records from a JSONL file. Deduplicates by (experiment_id, sample_idx) by default,
    """
    records = []
    if not Path(path).exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if deduplicate:
        deduped = {}
        for r in records:
            if r.get("experiment_id") is not None:
                deduped[record_key(r)] = r
            else:
                deduped[id(r)] = r
        return list(deduped.values())

    return records



def get_completed_ids(jsonl_path: Union[str, Path]) -> set:
    """
    Record keys (see record_key) already finalized in a JSONL file, for resumable runs.
    Errored, parse-failed and truncated records are NOT finalized, so they are retried on the next run.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return set()
    completed = set()
    for record in load_jsonl(path, deduplicate=True):
        if not record.get("experiment_id"):
            continue

        is_failed = (
            bool(record.get("error")) or
            record.get("constraint_type") == "parse_failure" or
            record.get("combined_label") == "PARSE_ERROR" or
            is_truncated(record)
        )

        if not is_failed:
            completed.add(record_key(record))

    return completed


def timestamp() -> str:
    """Get current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


# Grading

def grade_numeric_answer(final_answer: Any, reference: Any) -> dict:
    """
    Rule-judge grading of a visible answer against a reference value.
      is_correct              : PIH tolerance (config.ANSWER_TOLERANCE) - used in all PIH tables.
      is_correct_matscibench  : MatSciBench's official tolerance - kept for comparability with published numbers.
    """
    from evaluation.rule_judge import judge_num_answer
    from symbolic_engine import normalize_latex_number_formatting

    strict = judge_num_answer(
        normalize_latex_number_formatting(str(final_answer or "")), str(reference), multiple=False, rel_tol=ANSWER_TOLERANCE
    )
    official = judge_num_answer(final_answer, str(reference), multiple=False, rel_tol=MATSCIBENCH_TOLERANCE)
    return {
        "is_correct": strict["is_correct"],
        "is_correct_matscibench": official["is_correct"],
        "judge_reasoning": strict.get("judge_reasoning", ""),
    }


# Display Helpers

def progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> str:
    """Simple text progress bar."""
    frac = current / total if total > 0 else 0
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100
    return f"{prefix}|{bar}| {current}/{total} ({pct:.1f}%)"
