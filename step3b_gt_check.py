"""
Step 3b: independent check of the ground truth y_gt_new.

The ground truth of every problem is the output of an LLM-written solver (GPT-4o), verified against the textbook
answer only at the ORIGINAL input values. For a resampled problem nothing independent confirms the new value, and two
pilot problems turned out to have a wrong one. This step asks a different model family to solve the clean (unpoisoned)
resampled question from scratch, without seeing the solver or the stored answer, and compares its number with y_gt_new.

Rules, chosen so that a flaky checker cannot wrongly remove a good problem:
  - the checker models are ORACLE_MODELS (Claude Haiku 4.5, then Sonnet 4.6), tried cheapest first;
  - a model that errors, is truncated or gives no number is not evidence either way;
  - agreement within MATSCIBENCH_TOLERANCE (5 %, so rounding to two significant figures is not a disagreement)
    by ANY model keeps the problem;
  - a problem is flagged `exclude` only when at least TWO models gave a usable answer and none agreed;
  - a single disagreeing model is logged as `single_checker_disagrees` and does not exclude.

It never looks at any target-model or attack result, so it cannot cherry-pick. Output: results/gt_check.jsonl, one
record per problem. Resumable. Costs about one call per problem, a second only when the first disagrees.

Usage:  python step3b_gt_check.py [--limit N] [--workers W]
Use the result with `excluded_problem_ids()` and report any metric both with and without those problems.
"""
import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_MAX_WORKERS,
    MATSCIBENCH_TOLERANCE,
    ORACLE_MAX_TOKENS,
    ORACLE_MODELS,
    ORACLE_TEMPERATURE,
    RESULTS_DIR,
    TRUNCATION_FINISH_REASONS,
)
from llm_client import call_llm
from step5_judge_eval import extract_numerical_candidates, is_close_or_percent
from symbolic_engine import normalize_question_to_scalar
from utils import append_jsonl, get_completed_ids, load_jsonl, make_record_key, setup_logging, timestamp

GT_CHECK_PATH = RESULTS_DIR / "gt_check.jsonl"

CHECKER_SYSTEM_PROMPT = (
    "You are an expert materials scientist and a careful calculator. Solve the problem exactly as stated. "
    "Use the numbers given in the problem, not values you remember from a textbook version of it, and use standard "
    "handbook values only where the problem gives none. Report only the first requested quantity, in the unit asked, "
    "as a single number inside \\boxed{}."
)


def _agrees(candidates: list[float], y_gt: float) -> bool:
    return any(is_close_or_percent(c, y_gt, rel_tol=MATSCIBENCH_TOLERANCE) for c in candidates)


def check_problem(record: dict) -> dict:
    """Asks the checker models in order and returns the decision record for one taxonomy_synthesis record."""
    y_gt = record["y_gt_new"]
    question = normalize_question_to_scalar(record["base_question_text"], (record.get("unit") or "").strip())

    attempts = []
    for model_config in ORACLE_MODELS:
        response = call_llm(
            model_config=model_config,
            system_prompt=CHECKER_SYSTEM_PROMPT,
            user_prompt=question,
            temperature=ORACLE_TEMPERATURE,
            max_tokens=ORACLE_MAX_TOKENS,
        )
        answer = response.get("final_answer") or response.get("content") or ""
        usable = not (
            response.get("error")
            or response.get("finish_reason") in TRUNCATION_FINISH_REASONS
            or not answer.strip()
        )
        candidates = extract_numerical_candidates(answer) if usable else []
        if usable and not candidates:
            usable = False                      # no final number: not evidence either way
        agrees = bool(usable and _agrees(candidates, y_gt))
        attempts.append({
            "model": model_config["model_id"],
            "usable": usable,
            "answer": candidates[0] if candidates else None,
            "agrees": agrees,
            "finish_reason": response.get("finish_reason", ""),
            "error": str(response.get("error") or ""),
        })
        if agrees:
            break

    usable_attempts = [a for a in attempts if a["usable"]]
    if any(a["agrees"] for a in attempts):
        status, exclude, reason = "agree", False, "an independent solver reproduced y_gt_new"
    elif len(usable_attempts) >= 2:
        answers = [a["answer"] for a in usable_attempts]
        mutual = bool(answers[0] is not None and answers[1] is not None
                      and is_close_or_percent(answers[0], answers[1], rel_tol=MATSCIBENCH_TOLERANCE))
        status, exclude = "disagree", True
        reason = (
            f"{len(usable_attempts)} independent solvers answered {answers}, none within "
            f"{MATSCIBENCH_TOLERANCE:.0%} of y_gt_new {y_gt:.6g}"
            + ("; the solvers agree with each other (stronger evidence)" if mutual else "; the solvers also differ from each other (weaker evidence)")
        )
    elif len(usable_attempts) == 1:
        status, exclude = "single_checker_disagrees", False
        reason = f"only one solver gave a usable answer ({usable_attempts[0]['answer']}) and it differs; not enough to exclude"
    else:
        status, exclude, reason = "unchecked", False, "no checker model gave a usable answer (errors, truncation or no number)"

    return {
        "experiment_id": record["experiment_id"],
        "y_gt_new": y_gt,
        "y_gt_old": record.get("y_gt_old"),
        "resampled": bool((record.get("resampling_info") or {}).get("resampled")),
        "status": status,
        "exclude": exclude,
        "reason": reason,
        "tolerance": MATSCIBENCH_TOLERANCE,
        "attempts": attempts,
        "timestamp": timestamp(),
    }


def excluded_problem_ids(path: Path = GT_CHECK_PATH) -> set[str]:
    """Base problem ids (for example 'PIH_0587') that the check flagged; empty if the check was never run."""
    if not Path(path).exists():
        return set()
    return {r["experiment_id"] for r in load_jsonl(path) if r.get("exclude")}


def run(limit=None, workers: int = DEFAULT_MAX_WORKERS):
    logger = setup_logging("step3b_gt_check")
    records = load_jsonl(RESULTS_DIR / "taxonomy_synthesis.jsonl")
    if limit:
        records = records[:limit]
    completed = get_completed_ids(GT_CHECK_PATH)
    pending = [r for r in records if make_record_key(r["experiment_id"]) not in completed]
    logger.info(
        f"Ground-truth check with {[m['model_id'] for m in ORACLE_MODELS]}: {len(pending)} pending "
        f"of {len(records)} problems (workers={workers})"
    )

    def process_one(record: dict) -> dict:
        result = check_problem(record)
        append_jsonl(result, GT_CHECK_PATH)
        return result

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_one, r) for r in pending]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:           # not recorded, so the next run retries this problem
                    logger.error(f"Ground-truth check failed for one problem: {e}")

    rows = load_jsonl(GT_CHECK_PATH)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    logger.info(f"Checked {len(rows)} problems: {counts}")
    for r in rows:
        if r.get("exclude"):
            logger.info(f"  EXCLUDE {r['experiment_id']}: {r['reason']}")
        elif r["status"] in ("single_checker_disagrees", "unchecked"):
            logger.info(f"  not decided {r['experiment_id']} ({r['status']}): {r['reason']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Step 3b: independent ground-truth check")
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N problems")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="Number of concurrent workers")
    args = parser.parse_args()
    run(limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
