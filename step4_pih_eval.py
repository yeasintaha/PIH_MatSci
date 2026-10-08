import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PIH_DATASET_PATH,
    RESULTS_DIR,
    TARGET_MODELS,
    TARGET_TEMPERATURE,
    TARGET_MAX_TOKENS,
    DEFAULT_MAX_WORKERS,
)
from prompts import TARGET_SYSTEM_PROMPT, TARGET_QUESTION_TEMPLATE
from llm_client import call_llm
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


def run_pih_eval(model_config: dict, dataset: list, limit: int = None, workers: int = DEFAULT_MAX_WORKERS):
    model_name = model_config["display_name"]
    safe_name = model_name.replace(" ", "_").replace("/", "_").lower()
    output_path = RESULTS_DIR / f"pih_{safe_name}.jsonl"

    logger = setup_logging(f"step4_pih_{safe_name}")
    logger.info(f"Starting PIH evaluation: {model_name} (workers={workers})")

    completed = get_completed_ids(output_path)
    logger.info(f"Already completed: {len(completed)}")

    questions = dataset[:limit] if limit else dataset
    pending = [q for q in questions if q["experiment_id"] not in completed]
    logger.info(f"Pending: {len(pending)} questions")

    def process_one(question: dict) -> dict:
        eid = question["experiment_id"]
        user_prompt = TARGET_QUESTION_TEMPLATE.format(question=question["pih_question"])

        response = call_llm(
            model_config=model_config,
            system_prompt=TARGET_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=TARGET_TEMPERATURE,
            max_tokens=TARGET_MAX_TOKENS,
        )

        from evaluation.rule_judge import judge_num_answer
        judge_result = judge_num_answer(response.get("content", ""), str(question.get("answer", "")), multiple=False)
        is_correct = judge_result["is_correct"]

        record = {
            "experiment_id": eid,
            "qid": question["qid"],
            "model": model_name,
            "model_id": model_config["model_id"],
            "provider": model_config["provider"],
            "phase": "pih",
            "question_type": question["type"],
            "difficulty": question["difficulty_level"],
            "category": question["primary_category"],
            "constraint": question["constraint"],
            "constraint_type": question["constraint_type"],
            "subtlety_rating": question["subtlety_rating"],
            "ground_truth_answer": question["answer"],
            "predicted_number": None,
            "is_correct": is_correct,
            "judge_reasoning": judge_result.get("judge_reasoning", ""),
            "response_text": response["content"],
            "usage": response.get("usage", {}),
            "finish_reason": response.get("finish_reason", ""),
            "error": response.get("error"),
            "timestamp": timestamp(),
        }
        append_jsonl(record, output_path)
        return record

    if pending:
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_q = {executor.submit(process_one, q): q for q in pending}
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_q):
                    completed_count += 1
                    logger.info(progress_bar(completed_count, len(pending), prefix=f"[{model_name}] "))
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing question: {e}")
        else:
            for i, question in enumerate(pending):
                logger.info(progress_bar(i + 1, len(pending), prefix=f"[{model_name}] "))
                process_one(question)

    all_results = load_jsonl(output_path)
    total = len(all_results)
    correct = sum(1 for r in all_results if r["is_correct"])
    errors = sum(1 for r in all_results if r.get("error"))

    logger.info(f"PIH results ({model_name})")
    logger.info(f"Total: {total}")
    logger.info(f"Correct: {correct} ({100*correct/total:.1f}%)" if total else "Total: 0")
    logger.info(f"Wrong: {total - correct - errors}")
    logger.info(f"Errors: {errors}")

    baseline_path = RESULTS_DIR / f"baseline_{safe_name}.jsonl"
    if baseline_path.exists() and total > 0:
        baseline = load_jsonl(baseline_path)
        baseline_total = len(baseline)
        if baseline_total > 0:
            baseline_correct = sum(1 for r in baseline if r["is_correct"])
            drop = (baseline_correct / baseline_total - correct / total) * 100
            logger.info(f"Baseline accuracy: {100*baseline_correct/baseline_total:.1f}%")
            logger.info(f"PIH accuracy: {100*correct/total:.1f}%")
            logger.info(f"Accuracy drop: {drop:.1f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Step 4: PIH Evaluation")
    parser.add_argument("--model", type=str, help="Run only this model")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_MAX_WORKERS, help="Number of concurrent worker threads")
    args = parser.parse_args()

    dataset = load_json(PIH_DATASET_PATH)
    print(f"Loaded {len(dataset)} PIH questions")

    if args.model:
        models = [m for m in TARGET_MODELS if m["display_name"] == args.model]
        if not models:
            available = [m["display_name"] for m in TARGET_MODELS]
            print(f"Model '{args.model}' not found. Available: {available}")
            sys.exit(1)
    else:
        models = TARGET_MODELS

    for model_config in models:
        run_pih_eval(model_config, dataset, limit=args.limit, workers=args.workers)

    print("\nDoneee.")


if __name__ == "__main__":
    main()
