import argparse
import concurrent.futures
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLEAN_DATASET_PATH,
    PIH_DATASET_PATH,
    DATA_DIR,
    RESULTS_DIR,
    ATTACKER_MODEL,
    ATTACKER_TEMPERATURE,
    ATTACKER_MAX_TOKENS,
    DEFAULT_MAX_WORKERS,
)
from prompts import ATTACKER_SYSTEM_PROMPT, ATTACKER_TEMPLATE
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


def generate_constraints(dataset: list, limit: int = None, workers: int = DEFAULT_MAX_WORKERS):
    logger = setup_logging("step3_attack")

    work_path = RESULTS_DIR / "attack_synthesis.jsonl"
    completed = get_completed_ids(work_path)
    logger.info(f"Already completed: {len(completed)} constraints")

    questions = dataset[:limit] if limit else dataset
    pending = [q for q in questions if q["experiment_id"] not in completed]
    logger.info(f"Generating constraints for {len(pending)} questions using {ATTACKER_MODEL['display_name']} (workers={workers})")

    failed_lock = threading.Lock()
    failed_count = [0]

    def process_one(question: dict) -> dict:
        eid = question["experiment_id"]
        user_prompt = ATTACKER_TEMPLATE.format(
            question=question["question"],
            solution=question["solution"],
            answer=question["answer"],
        )

        response = call_llm(
            model_config=ATTACKER_MODEL,
            system_prompt=ATTACKER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=ATTACKER_TEMPERATURE,
            max_tokens=ATTACKER_MAX_TOKENS,
        )

        parsed = extract_json_from_response(response["content"]) if response["content"] else None

        if parsed and "constraint" in parsed:
            record = {
                "experiment_id": eid,
                "qid": question["qid"],
                "invariant_identified": parsed.get("invariant_identified", ""),
                "constraint": parsed["constraint"],
                "constraint_type": parsed.get("constraint_type", "unknown"),
                "subtlety_rating": parsed.get("subtlety_rating", 0),
                "reasoning": parsed.get("reasoning", ""),
                "attacker_raw_response": response["content"],
                "usage": response.get("usage", {}),
                "timestamp": timestamp(),
                "error": None,
            }
        else:
            with failed_lock:
                failed_count[0] += 1
            logger.warning(f"  Failed to parse constraint for {eid}. Raw: {response.get('content', '')[:200]}")
            record = {
                "experiment_id": eid,
                "qid": question["qid"],
                "invariant_identified": "",
                "constraint": "",
                "constraint_type": "parse_failure",
                "subtlety_rating": 0,
                "reasoning": "",
                "attacker_raw_response": response.get("content", ""),
                "usage": response.get("usage", {}),
                "timestamp": timestamp(),
                "error": response.get("error", "Failed to parse JSON from response"),
            }

        append_jsonl(record, work_path)
        return record

    if pending:
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_q = {executor.submit(process_one, q): q for q in pending}
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_q):
                    completed_count += 1
                    logger.info(progress_bar(completed_count, len(pending), prefix="[Attack] "))
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error processing question: {e}")
        else:
            for i, question in enumerate(pending):
                logger.info(progress_bar(i + 1, len(pending), prefix="[Attack] "))
                process_one(question)

    logger.info(f"\nConstraint generation complete. Failed: {failed_count[0]}/{len(pending)}")
    return work_path


def build_pih_dataset(dataset: list, constraint_path: Path):
    logger = setup_logging("step3_build_pih")

    constraints = load_jsonl(constraint_path)
    constraint_map = {c["experiment_id"]: c for c in constraints}

    pih_data = []
    skipped = 0

    for question in dataset:
        eid = question["experiment_id"]
        constraint_record = constraint_map.get(eid)

        if not constraint_record or not constraint_record.get("constraint"):
            skipped += 1
            continue

        pih_question = (
            f"{question['question']}\n\n"
            f"Additional constraint: {constraint_record['constraint']}"
        )

        pih_entry = {
            **question,
            "pih_question": pih_question,
            "constraint": constraint_record["constraint"],
            "constraint_type": constraint_record["constraint_type"],
            "subtlety_rating": constraint_record["subtlety_rating"],
            "invariant_identified": constraint_record["invariant_identified"],
        }
        pih_data.append(pih_entry)

    save_json(pih_data, PIH_DATASET_PATH)
    logger.info(f"PIH dataset: {len(pih_data)} questions saved to {PIH_DATASET_PATH}")
    logger.info(f"Skipped (no valid constraint): {skipped}")

    from collections import Counter
    type_dist = Counter(q["constraint_type"] for q in pih_data)
    logger.info("Constraint Type Distribution")
    for ct, count in type_dist.most_common():
        logger.info(f"  {ct}: {count}")

    subtlety_dist = Counter(q["subtlety_rating"] for q in pih_data)
    logger.info("Subtlety Rating Distribution")
    for sr, count in sorted(subtlety_dist.items()):
        logger.info(f"  Rating {sr}: {count}")

    return pih_data


def review_constraints(constraint_path: Path, n: int = 20):
    constraints = load_jsonl(constraint_path)
    valid = [c for c in constraints if c.get("constraint")]

    print(f"Constraint Review ({len(valid)} valid): {min(n, len(valid))} examples   ")

    import random
    sample = random.sample(valid, min(n, len(valid)))

    for i, c in enumerate(sample, 1):
        print(f"{i}. {c['experiment_id']}")
        print(f"Invariant: {c['invariant_identified']}")
        print(f"Constraint: {c['constraint']}")
        print(f"Type: {c['constraint_type']}")
        print(f"Subtlety: {c['subtlety_rating']}/5")
        print(f"Reasoning: {c['reasoning']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Step 3: Attack Synthesis")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_MAX_WORKERS, help="Number of concurrent worker threads")
    parser.add_argument("--review", action="store_true", help="Review generated constraints")
    parser.add_argument("--build-only", action="store_true", help="Skip generation, just build PIH dataset from existing constraints")
    args = parser.parse_args()

    dataset = load_json(CLEAN_DATASET_PATH)
    print(f"Loaded {len(dataset)} clean questions")

    constraint_path = RESULTS_DIR / "attack_synthesis.jsonl"

    if args.review:
        review_constraints(constraint_path)
        return

    if not args.build_only:
        constraint_path = generate_constraints(dataset, limit=args.limit, workers=args.workers)
    
    target_dataset = dataset[:args.limit] if args.limit else dataset
    build_pih_dataset(target_dataset, constraint_path)

    print("done")


if __name__ == "__main__":
    main()
