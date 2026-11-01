import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLEAN_DATASET_PATH,
    PIH_DATASET_PATH,
    RESULTS_DIR,
    TARGET_MODELS,
    TARGET_MAX_TOKENS,
    TARGET_SAMPLES_PER_ITEM,
    DEFAULT_MAX_WORKERS,
)
from prompts import TARGET_SYSTEM_PROMPT, TARGET_QUESTION_TEMPLATE
from llm_client import call_target_model, target_sampling_params
from utils import (
    append_jsonl,
    base_experiment_id,
    get_completed_ids,
    grade_numeric_answer,
    is_truncated,
    load_json,
    load_jsonl,
    make_record_key,
    save_json,
    setup_logging,
    timestamp,
    progress_bar,
)


def run_baseline(
    model_config: dict,
    dataset: list,
    limit: int = None,
    workers: int = DEFAULT_MAX_WORKERS,
    samples: int = TARGET_SAMPLES_PER_ITEM,
    max_tokens: int = TARGET_MAX_TOKENS,
):
    model_name = model_config["display_name"]
    safe_name = model_name.replace(" ", "_").replace("/", "_").lower()
    output_path = RESULTS_DIR / f"baseline_{safe_name}.jsonl"
    sampling = target_sampling_params(model_config)

    logger = setup_logging(f"step2_baseline_{safe_name}")
    logger.info(f"Starting baseline evaluation: {model_name} (workers={workers}, samples={samples}, sampling={sampling}, max_tokens={max_tokens})")
    logger.info(f"Output: {output_path}")

    completed = get_completed_ids(output_path)
    logger.info(f"Already completed: {len(completed)}")

    questions = dataset[:limit] if limit else dataset
    pending = [
        (q, s) for q in questions for s in range(samples)
        if make_record_key(q["experiment_id"], s) not in completed
    ]
    logger.info(f"Remaining: {len(pending)}")

    def process_one(item: tuple) -> dict:
        question, sample_idx = item
        eid = question["experiment_id"]

        q_text = question["question"]
        unit = question.get("unit", "").strip()
        num_ans = question.get("number_of_answers", "")

        # MatSciBench's official unit hint (clean questions carry number_of_answers from step1).
        if unit:
            if num_ans == "single":
                q_text += f" The unit of the answer is {unit}."
            elif num_ans == "multiple":
                q_text += f" The units of each required answer are {unit}, respectively."

        user_prompt = TARGET_QUESTION_TEMPLATE.format(question=q_text)
        response = call_target_model(model_config, TARGET_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens)

        final_answer = response.get("final_answer", response.get("content", ""))
        grade = grade_numeric_answer(final_answer, question.get("answer", ""))

        record = {
            "experiment_id": eid,
            "sample_idx": sample_idx,
            "qid": question["qid"],
            "model": model_name,
            "model_id": model_config["model_id"],
            "provider": model_config["provider"],
            "sampling": sampling,
            "phase": "baseline",
            "question_type": question["type"],
            "difficulty": question["difficulty_level"],
            "category": question["primary_category"],
            "ground_truth_answer": question["answer"],
            "predicted_number": None,
            "is_correct": grade["is_correct"],
            "is_correct_matscibench": grade["is_correct_matscibench"],
            "judge_reasoning": grade["judge_reasoning"],
            "response_text": response.get("content", ""),
            "think_trace": response.get("think_trace", ""),
            "final_answer": final_answer,
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
                future_to_q = {executor.submit(process_one, item): item for item in pending}
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_q):
                    completed_count += 1
                    logger.info(progress_bar(completed_count, len(pending), prefix=f"[{model_name}] "))
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing question: {e}")
        else:
            for i, item in enumerate(pending):
                logger.info(progress_bar(i + 1, len(pending), prefix=f"[{model_name}] "))
                process_one(item)

    all_results = load_jsonl(output_path)
    total = len(all_results)
    errors = sum(1 for r in all_results if r.get("error"))
    truncated = sum(1 for r in all_results if is_truncated(r) and not r.get("error"))
    scored = [r for r in all_results if not r.get("error") and not is_truncated(r)]
    correct = sum(1 for r in scored if r["is_correct"])

    logger.info(f"{model_name}")
    logger.info(f"Total records: {total} | Errors: {errors} | Truncated: {truncated} | Scored: {len(scored)}")
    if scored:
        logger.info(f"Correct (scored, {len(scored)}): {correct} ({100*correct/len(scored):.1f}%)")

    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in scored if r["difficulty"] == diff]
        if subset:
            diff_correct = sum(1 for r in subset if r["is_correct"])
            logger.info(f"  {diff:8s}: {diff_correct}/{len(subset)} ({100*diff_correct/len(subset):.1f}%)")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Step 2: Baseline Evaluation")
    parser.add_argument("--model", type=str, help="Run only this model")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_MAX_WORKERS, help="Number of concurrent worker threads")
    parser.add_argument("--samples", type=int, default=TARGET_SAMPLES_PER_ITEM, help="Samples per question (sample_idx 0..k-1)")
    parser.add_argument("--max-tokens", type=int, default=TARGET_MAX_TOKENS, help="Generation ceiling (raise to re-query truncated items)")
    parser.add_argument("--ids-from-pih", action="store_true",
                        help="Evaluate only the base problems present in data/pih_dataset.json (same scope as the PIH run)")
    args = parser.parse_args()

    dataset = load_json(CLEAN_DATASET_PATH)
    print(f"Loaded {len(dataset)} questions from clean dataset")
    if args.ids_from_pih:
        pih_bases = {base_experiment_id(q) for q in load_json(PIH_DATASET_PATH)}
        dataset = [q for q in dataset if q["experiment_id"] in pih_bases]
        print(f"Restricted to {len(dataset)} base problems present in {PIH_DATASET_PATH.name}")

    if args.model:
        models = [m for m in TARGET_MODELS if m["display_name"] == args.model]
        if not models:
            print(f"Model '{args.model}' not found. Available: {[m['display_name'] for m in TARGET_MODELS]}")
            sys.exit(1)
    else:
        models = TARGET_MODELS

    all_summaries = {}
    for model_config in models:
        results = run_baseline(
            model_config, dataset, limit=args.limit, workers=args.workers,
            samples=args.samples, max_tokens=args.max_tokens,
        )
        scored = [r for r in results if not r.get("error") and not is_truncated(r)]
        correct = sum(1 for r in scored if r["is_correct"])
        all_summaries[model_config["display_name"]] = {
            "total": len(results),
            "scored": len(scored),
            "truncated": sum(1 for r in results if is_truncated(r) and not r.get("error")),
            "correct": correct,
            "accuracy": round(correct / len(scored) * 100, 2) if scored else 0,
        }

    save_json(all_summaries, RESULTS_DIR / "baseline_summary.json")
    for model, stats in all_summaries.items():
        print(f"  {model:25s}: {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['scored']} scored, {stats['truncated']} truncated)")


if __name__ == "__main__":
    main()
