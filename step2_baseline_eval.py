import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLEAN_DATASET_PATH,
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


def run_baseline(model_config: dict, dataset: list, limit: int = None, workers: int = DEFAULT_MAX_WORKERS):
    model_name = model_config["display_name"]
    safe_name = model_name.replace(" ", "_").replace("/", "_").lower()
    output_path = RESULTS_DIR / f"baseline_{safe_name}.jsonl"

    logger = setup_logging(f"step2_baseline_{safe_name}")
    logger.info(f"Starting baseline evaluation: {model_name} (workers={workers})")
    logger.info(f"Output: {output_path}")

    completed = get_completed_ids(output_path)
    logger.info(f"Already completed: {len(completed)}")

    questions = dataset[:limit] if limit else dataset
    pending = [q for q in questions if q["experiment_id"] not in completed]
    logger.info(f"Remaining: {len(pending)}")

    def process_one(question: dict) -> dict:
        eid = question["experiment_id"]
        user_prompt = TARGET_QUESTION_TEMPLATE.format(question=question["question"])
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
            "phase": "baseline",
            "question_type": question["type"],
            "difficulty": question["difficulty_level"],
            "category": question["primary_category"],
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

    logger.info(f"{model_name}")
    logger.info(f"Total: {total}")
    logger.info(f"Correct: {correct} ({100*correct/total:.1f}%)" if total else "Total: 0")
    logger.info(f"Wrong: {total - correct - errors} ({100*(total-correct-errors)/total:.1f}%)" if total else "")
    logger.info(f"Errors: {errors}")

    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in all_results if r["difficulty"] == diff]
        if subset:
            diff_correct = sum(1 for r in subset if r["is_correct"])
            logger.info(f"  {diff:8s}: {diff_correct}/{len(subset)} ({100*diff_correct/len(subset):.1f}%)")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Step 2: Baseline Evaluation")
    parser.add_argument("--model", type=str, help="Run only this model")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_MAX_WORKERS, help="Number of concurrent worker threads")
    args = parser.parse_args()

    dataset = load_json(CLEAN_DATASET_PATH)
    print(f"Loaded {len(dataset)} questions from clean dataset")

    if args.model:
        models = [m for m in TARGET_MODELS if m["display_name"] == args.model]
        if not models:
            print(f"Model '{args.model}' not found. Available: {[m['display_name'] for m in TARGET_MODELS]}")
            sys.exit(1)
    else:
        models = TARGET_MODELS

    all_summaries = {}
    for model_config in models:
        results = run_baseline(model_config, dataset, limit=args.limit, workers=args.workers)
        correct = sum(1 for r in results if r["is_correct"])
        all_summaries[model_config["display_name"]] = {
            "total": len(results),
            "correct": correct,
            "accuracy": round(correct / len(results) * 100, 2) if results else 0,
        }

    save_json(all_summaries, RESULTS_DIR / "baseline_summary.json")
    for model, stats in all_summaries.items():
        print(f"  {model:25s}: {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()
