import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
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
    setup_logging,
    timestamp,
    progress_bar,
)

CLEAN_TYPES = {"type_0_clean", "type_0_clean_resampled", "clean"}


def run_pih_eval(
    model_config: dict,
    dataset: list,
    limit: int = None,
    workers: int = DEFAULT_MAX_WORKERS,
    samples: int = TARGET_SAMPLES_PER_ITEM,
    max_tokens: int = TARGET_MAX_TOKENS,
):
    model_name = model_config["display_name"]
    safe_name = model_name.replace(" ", "_").replace("/", "_").lower()
    output_path = RESULTS_DIR / f"pih_{safe_name}.jsonl"
    sampling = target_sampling_params(model_config)

    logger = setup_logging(f"step4_pih_{safe_name}")
    logger.info(f"Starting PIH evaluation: {model_name} (workers={workers}, samples={samples}, sampling={sampling}, max_tokens={max_tokens})")

    completed = get_completed_ids(output_path)
    logger.info(f"Already completed: {len(completed)}")

    questions = dataset[:limit] if limit else dataset
    pending = [
        (q, s) for q in questions for s in range(samples)
        if make_record_key(q["experiment_id"], s) not in completed
    ]
    logger.info(f"Pending: {len(pending)} generations (truncated/errored records are re-queried)")

    def process_one(item: tuple) -> dict:
        question, sample_idx = item
        eid = question["experiment_id"]

        # pih_question already carries the scalar/unit lock from normalize_question_to_scalar (step3).
        user_prompt = TARGET_QUESTION_TEMPLATE.format(question=question["pih_question"])
        response = call_target_model(model_config, TARGET_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens)

        think_trace = response.get("think_trace", "")
        final_answer = response.get("final_answer", response.get("content", ""))
        reference = question.get("y_gt_new", question.get("answer"))
        grade = grade_numeric_answer(final_answer, reference)

        record = {
            "experiment_id": eid,
            "sample_idx": sample_idx,
            "base_experiment_id": base_experiment_id(question),
            "qid": question["qid"],
            "model": model_name,
            "model_id": model_config["model_id"],
            "provider": model_config["provider"],
            "sampling": sampling,
            "max_tokens": max_tokens,
            "phase": "pih",
            "question_type": question["type"],
            "difficulty": question["difficulty_level"],
            "category": question["primary_category"],
            "constraint": question["constraint"],
            "constraint_type": question.get("constraint_type", ""),
            "constraint_name": question.get("constraint_name", ""),
            "sub_category": question.get("sub_category", ""),
            "bound_kind": question.get("bound_kind", ""),
            "violated_invariant": question.get("violated_invariant", ""),
            "subtlety_rating": question.get("subtlety_rating", 3),
            "ground_truth_answer": reference,
            "y_gt_new": reference,
            "y_gt_old": question.get("y_gt_old", question.get("answer")),
            "y_syc_new": question.get("y_syc_new"),
            "oracle_status": question.get("oracle_status", ""),
            "memorization_trap_active": question.get("memorization_trap_active", False),
            "predicted_number": None,
            "is_correct": grade["is_correct"],
            "is_correct_matscibench": grade["is_correct_matscibench"],
            "judge_reasoning": grade["judge_reasoning"],
            "response_text": response.get("content", ""),
            "think_trace": think_trace,
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
    errors = sum(1 for r in all_results if r.get("error"))
    truncated = sum(1 for r in all_results if is_truncated(r) and not r.get("error"))
    no_trace = sum(1 for r in all_results if not r.get("error") and not (r.get("think_trace") or "").strip())
    scored = [r for r in all_results if not r.get("error") and not is_truncated(r)]
    adv = [r for r in scored if r.get("constraint_type") not in CLEAN_TYPES]
    clean = [r for r in scored if r.get("constraint_type") in CLEAN_TYPES]

    logger.info(f"PIH results ({model_name})")
    logger.info(f"Total records: {len(all_results)} | Errors: {errors} | Truncated: {truncated} | Empty think trace: {no_trace}")
    if adv:
        adv_correct = sum(1 for r in adv if r["is_correct"])
        logger.info(f"Adversarial accuracy vs y_gt_new (scored): {adv_correct}/{len(adv)} ({100*adv_correct/len(adv):.1f}%)")
    if clean:
        clean_correct = sum(1 for r in clean if r["is_correct"])
        logger.info(f"Clean-control accuracy vs y_gt_new (scored): {clean_correct}/{len(clean)} ({100*clean_correct/len(clean):.1f}%)")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Step 4: PIH Evaluation")
    parser.add_argument("--model", type=str, help="Run only this model")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_MAX_WORKERS, help="Number of concurrent worker threads")
    parser.add_argument("--samples", type=int, default=TARGET_SAMPLES_PER_ITEM, help="Samples per question (sample_idx 0..k-1)")
    parser.add_argument("--max-tokens", type=int, default=TARGET_MAX_TOKENS, help="Generation ceiling (raise to re-query truncated items)")
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
        run_pih_eval(
            model_config, dataset, limit=args.limit, workers=args.workers,
            samples=args.samples, max_tokens=args.max_tokens,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
