import argparse
import concurrent.futures
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PIH_DATASET_PATH,
    RESULTS_DIR,
    TARGET_MODELS,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    JUDGE_MAX_TOKENS,
    DEFAULT_MAX_WORKERS,
)
from prompts import JUDGE_SYSTEM_PROMPT, JUDGE_TEMPLATE
from llm_client import call_llm, extract_json_from_response
from utils import (
    load_json,
    save_json,
    append_jsonl,
    load_jsonl,
    get_completed_ids,
    setup_logging,
    timestamp,
    progress_bar,
)


def judge_model_responses(model_config: dict, pih_dataset: list, limit: int = None, workers: int = DEFAULT_MAX_WORKERS):
    model_name = model_config["display_name"]
    safe_name = model_name.replace(" ", "_").replace("/", "_").lower()

    pih_response_path = RESULTS_DIR / f"pih_{safe_name}.jsonl"
    if not pih_response_path.exists():
        print(f"No PIH responses found for {model_name} at {pih_response_path}")
        print("Run step4_pih_eval.py first.")
        return []

    pih_responses = load_jsonl(pih_response_path)
    response_map = {r["experiment_id"]: r for r in pih_responses}

    judge_output_path = RESULTS_DIR / f"judge_{safe_name}.jsonl"
    completed = get_completed_ids(judge_output_path)

    logger = setup_logging(f"step5_judge_{safe_name}")
    logger.info(f"Judging {model_name} responses with {JUDGE_MODEL['display_name']} (workers={workers})")
    logger.info(f"PIH responses available: {len(pih_responses)}")
    logger.info(f"Already judged: {len(completed)}")

    q_map = {q["experiment_id"]: q for q in pih_dataset}

    pending = [
        r for r in pih_responses
        if r["experiment_id"] not in completed
        and r["experiment_id"] in q_map
        and not r.get("error")
    ]
    if limit:
        pending = pending[:limit]
    logger.info(f"Pending judgments: {len(pending)}")

    failed_lock = threading.Lock()
    parse_failures = [0]

    def process_one(response_record: dict) -> dict:
        eid = response_record["experiment_id"]
        question = q_map[eid]

        user_prompt = JUDGE_TEMPLATE.format(
            clean_question=question["question"],
            correct_answer=question["answer"],
            constraint=question["constraint"],
            constraint_type=question["constraint_type"],
            response=response_record["response_text"],
        )

        judge_response = call_llm(
            model_config=JUDGE_MODEL,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
        )

        parsed = extract_json_from_response(judge_response["content"]) if judge_response["content"] else None

        if parsed and "detection_score" in parsed:
            record = {
                "experiment_id": eid,
                "qid": question["qid"],
                "target_model": model_name,
                "judge_model": JUDGE_MODEL["display_name"],
                "detection_score": parsed["detection_score"],
                "correctness_score": parsed["correctness_score"],
                "combined_label": parsed.get("combined_label", "UNKNOWN"),
                "failure_mode": parsed.get("failure_mode", "unknown"),
                "justification": parsed.get("justification", ""),
                "constraint_type": question["constraint_type"],
                "subtlety_rating": question["subtlety_rating"],
                "difficulty": question["difficulty_level"],
                "category": question["primary_category"],
                "judge_raw_response": judge_response["content"],
                "judge_usage": judge_response.get("usage", {}),
                "timestamp": timestamp(),
                "error": None,
            }
        else:
            with failed_lock:
                parse_failures[0] += 1
            logger.warning(f"  Failed to parse judge response for {eid}")
            record = {
                "experiment_id": eid,
                "qid": question["qid"],
                "target_model": model_name,
                "judge_model": JUDGE_MODEL["display_name"],
                "detection_score": -1,
                "correctness_score": -1,
                "combined_label": "PARSE_ERROR",
                "failure_mode": "parse_error",
                "justification": "",
                "constraint_type": question["constraint_type"],
                "subtlety_rating": question["subtlety_rating"],
                "difficulty": question["difficulty_level"],
                "category": question["primary_category"],
                "judge_raw_response": judge_response.get("content", ""),
                "judge_usage": judge_response.get("usage", {}),
                "timestamp": timestamp(),
                "error": judge_response.get("error", "Failed to parse judge JSON"),
            }

        append_jsonl(record, judge_output_path)
        return record

    if pending:
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_r = {executor.submit(process_one, r): r for r in pending}
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_r):
                    completed_count += 1
                    logger.info(progress_bar(completed_count, len(pending), prefix=f"[Judge {model_name}] "))
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error judging item: {e}")
        else:
            for i, response_record in enumerate(pending):
                logger.info(progress_bar(i + 1, len(pending), prefix=f"[Judge {model_name}] "))
                process_one(response_record)

    all_judged = load_jsonl(judge_output_path)
    valid = [r for r in all_judged if r["detection_score"] >= 0]

    from collections import Counter
    labels = Counter(r["combined_label"] for r in valid)
    modes = Counter(r["failure_mode"] for r in valid)

    logger.info(f"{model_name}")
    logger.info(f"Total judged: {len(valid)} (parse failures: {parse_failures[0]})")
    logger.info(f"\nCombined Labels:")
    for label, count in labels.most_common():
        logger.info(f"{label:15s}: {count:4d} ({100*count/len(valid):.1f}%)")
    logger.info(f"\nFailure Modes:")
    for mode, count in modes.most_common():
        logger.info(f"{mode:25s}: {count:4d} ({100*count/len(valid):.1f}%)")

    sycophantic = sum(1 for r in valid if r["combined_label"] == "SYCOPHANTIC")
    solved = sum(1 for r in valid if r["combined_label"] == "SOLVED")
    detected = sum(1 for r in valid if r["combined_label"] == "DETECTED")
    logger.info(f"\nSycophancy Rate (ASR): {100*sycophantic/len(valid):.1f}%")
    logger.info(f"Solved Rate: {100*solved/len(valid):.1f}%")
    logger.info(f"Detected Rate: {100*detected/len(valid):.1f}%")

    return all_judged


def main():
    parser = argparse.ArgumentParser(description="Step 5: Judge Evaluation")
    parser.add_argument("--model", type=str, help="Judge only this target model")
    parser.add_argument("--limit", type=int, help="Limit number of judgments")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_MAX_WORKERS, help="Number of concurrent worker threads")
    args = parser.parse_args()

    pih_dataset = load_json(PIH_DATASET_PATH)
    print(f"Loaded {len(pih_dataset)} PIH questions")

    if args.model:
        models = [m for m in TARGET_MODELS if m["display_name"] == args.model]
        if not models:
            available = [m["display_name"] for m in TARGET_MODELS]
            print(f"Model '{args.model}' not found. Available: {available}")
            sys.exit(1)
    else:
        models = TARGET_MODELS

    all_summaries = {}
    for model_config in models:
        results = judge_model_responses(model_config, pih_dataset, limit=args.limit, workers=args.workers)
        if results:
            valid = [r for r in results if r["detection_score"] >= 0]
            sycophantic = sum(1 for r in valid if r["combined_label"] == "SYCOPHANTIC")
            solved = sum(1 for r in valid if r["combined_label"] == "SOLVED")
            detected = sum(1 for r in valid if r["combined_label"] == "DETECTED")
            all_summaries[model_config["display_name"]] = {
                "total": len(valid),
                "sycophantic": sycophantic,
                "solved": solved,
                "detected": detected,
                "asr": round(sycophantic / len(valid) * 100, 2) if valid else 0,
                "solved_rate": round(solved / len(valid) * 100, 2) if valid else 0,
            }

    save_json(all_summaries, RESULTS_DIR / "judge_summary.json")

    for model, s in all_summaries.items():
        print(f"{model:25s} {s['asr']:7.1f}% {s['solved_rate']:7.1f}% "
              f"{100*s['detected']/s['total']:7.1f}% {s['total']:5d}")


if __name__ == "__main__":
    main()
