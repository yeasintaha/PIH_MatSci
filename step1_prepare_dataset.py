import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import RAW_DATASET_PATH, CLEAN_DATASET_PATH, DATA_DIR
from utils import load_json, save_json, setup_logging


def prepare_dataset():
    logger = setup_logging("step1_prepare")

    logger.info(f"Loading raw dataset from {RAW_DATASET_PATH}")
    raw_data = load_json(RAW_DATASET_PATH)
    logger.info(f"Raw dataset: {len(raw_data)} questions")

    has_image = [q for q in raw_data if q.get("image") and len(q["image"]) > 0]
    no_image = [q for q in raw_data if not q.get("image") or len(q["image"]) == 0]
    logger.info(f"Questions with images: {len(has_image)}")
    logger.info(f"Questions without images: {len(no_image)}")

    clean_data = []
    for idx, q in enumerate(no_image):
        record = {
            "experiment_id": f"PIH_{idx:04d}",
            "qid": q["qid"],
            "type": q["type"],
            "question": q["question"],
            "solution": q["solution"],
            "answer": q["answer"],
            "unit": q.get("unit", ""),
            "number_of_answers": q.get("number_of_answers", "single"),
            "difficulty_level": q.get("difficulty_level", ""),
            "primary_category": q.get("primary_category", ""),
            "materials": q.get("Materials", ""),
            "properties": q.get("Properties", ""),
            "structures": q.get("Structures", ""),
            "fundamental_mechanisms": q.get("Fundamental Mechanisms", ""),
            "processes": q.get("Processes", ""),
            "failure_mechanisms": q.get("Failure Mechanisms", ""),
            "source": q.get("source", ""),
            "original_qid": q.get("original_qid", ""),
        }
        clean_data.append(record)

    logger.info(f"Clean dataset: {len(clean_data)} text-only questions")

    from collections import Counter

    types = Counter(q["type"] for q in clean_data)
    difficulties = Counter(q["difficulty_level"] for q in clean_data)
    categories = Counter(q["primary_category"] for q in clean_data)

    logger.info("Type Distribution")
    for t, c in types.most_common():
        logger.info(f"  {t}: {c}")

    logger.info("Difficulty Distribution")
    for d, c in difficulties.most_common():
        logger.info(f"  {d}: {c}")

    for cat, c in categories.most_common(10):
        logger.info(f"  {cat}: {c}")

    save_json(clean_data, CLEAN_DATASET_PATH)
    logger.info(f"Saved clean dataset to {CLEAN_DATASET_PATH}")

    image_data = []
    for idx, q in enumerate(has_image):
        record = {
            "experiment_id": f"PIH_IMG_{idx:04d}",
            "qid": q["qid"],
            "type": q["type"],
            "question": q["question"],
            "solution": q["solution"],
            "answer": q["answer"],
            "unit": q.get("unit", ""),
            "image_count": len(q["image"]),
            "difficulty_level": q.get("difficulty_level", ""),
            "primary_category": q.get("primary_category", ""),
            "source": q.get("source", ""),
        }
        image_data.append(record)
    save_json(image_data, DATA_DIR / "matscibench_image_subset.json")
    logger.info(f"Saved image subset ({len(image_data)} questions) for reference")

    logger.info("doneee")
    return clean_data


if __name__ == "__main__":
    prepare_dataset()
