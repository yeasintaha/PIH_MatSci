import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    missing = []
    for pkg, import_name in [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("python-dotenv", "dotenv"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def check_api_keys():
    from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, FIREWORKS_API_KEY

    keys = {
        "OpenAI": bool(OPENAI_API_KEY),
        "Anthropic": bool(ANTHROPIC_API_KEY),
        "OpenRouter": bool(OPENROUTER_API_KEY),
        "Fireworks": bool(FIREWORKS_API_KEY),
        "Google": bool(GOOGLE_API_KEY),
    }

    print("\nAPI Key Status:")
    for name, present in keys.items():
        status = "OK: configured" if present else "NOT configured"
        print(f"  {name:12s}: {status}")

    if not any(keys.values()):
        print("\nNo API keys found! Add them to .env file:")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="PIH Research Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  step1  - Dataset preparation (filter, assign IDs)
  step2  - Baseline evaluation (clean questions -> target LLMs)
  step3  - Attack synthesis (generate adversarial constraints)
  step4  - PIH evaluation (adversarial questions -> target LLMs)
  step5  - Judge evaluation (grade PIH responses)
  step6  - Human validation sample preparation (no LLM calls)
  step7  - Mitigation experiments (requires --defense)
  step8  - Analysis & reporting (no LLM calls)
  check  - Check dependencies and API keys
        """,
    )
    parser.add_argument("step", choices=["step1", "step2", "step3", "step4", "step5",
                                         "step6", "step7", "step8", "check", "all"])
    parser.add_argument("--model", type=str, help="Target model name")
    parser.add_argument("--limit", type=int, help="Limit questions (for testing)")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of concurrent worker threads")
    parser.add_argument("--defense", type=str, choices=["A", "B", "C", "all"], help="Mitigation defense (required for step7)")
    args, remaining = parser.parse_known_args()

    if args.step == "check":
        check_dependencies()
        check_api_keys()
        return

    if not check_dependencies():
        sys.exit(1)

    needs_api = args.step in {"step2", "step3", "step4", "step5", "step7", "all"}
    if needs_api and not check_api_keys():
        sys.exit(1)

    step_args = []
    if args.model:
        step_args.extend(["--model", args.model])
    if args.limit is not None:
        step_args.extend(["--limit", str(args.limit)])
    if args.workers is not None:
        step_args.extend(["--workers", str(args.workers)])
    if args.defense:
        step_args.extend(["--defense", args.defense])
    step_args.extend(remaining)

    if args.step == "step1" or args.step == "all":
        print("\n" + "=" * 60)
        print("STEP 1: Dataset Preparation")
        print("=" * 60)
        sys.argv = ["step1_prepare_dataset.py"]
        from step1_prepare_dataset import prepare_dataset
        prepare_dataset()

    if args.step == "step2" or args.step == "all":
        print("\n" + "=" * 60)
        print("STEP 2: Baseline Evaluation")
        print("=" * 60)
        sys.argv = ["step2_baseline_eval.py"] + step_args
        from step2_baseline_eval import main as step2_main
        step2_main()

    if args.step == "step3" or args.step == "all":
        print("\n" + "=" * 60)
        print("STEP 3: Taxonomy Attack Synthesis (Decompose -> Perturb -> Recompute)")
        print("=" * 60)
        sys.argv = ["step3_taxonomy_synthesis.py"] + step_args
        from step3_taxonomy_synthesis import main as step3_main
        step3_main()

    if args.step == "step4" or args.step == "all":
        print("\n" + "=" * 60)
        print("STEP 4: PIH Evaluation")
        print("=" * 60)
        sys.argv = ["step4_pih_eval.py"] + step_args
        from step4_pih_eval import main as step4_main
        step4_main()

    if args.step == "step5" or args.step == "all":
        print("\n" + "=" * 60)
        print("STEP 5: Judge Evaluation")
        print("=" * 60)
        sys.argv = ["step5_judge_eval.py"] + step_args
        from step5_judge_eval import main as step5_main
        step5_main()

    if args.step == "step6" or args.step == "all":
        print("\n" + "=" * 60)
        print("STEP 6: Human Validation Sample")
        print("=" * 60)
        sys.argv = ["step6_human_validation.py"] + step_args
        from step6_human_validation import main as step6_main
        step6_main()

    if args.step == "step7":
        print("\n" + "=" * 60)
        print("STEP 7: Mitigation Experiments")
        print("=" * 60)
        if not args.defense:
            print("Error: --defense is required for step7. Use A, B, C, or all.")
            sys.exit(1)
        sys.argv = ["step7_mitigation_eval.py"] + step_args
        from step7_mitigation_eval import main as step7_main
        step7_main()

    if args.step == "step8" or args.step == "all":
        print("\n" + "=" * 60)
        print("STEP 8: Analysis & Reporting")
        print("=" * 60)
        sys.argv = ["step8_analysis.py"] + step_args
        from step8_analysis import main as step8_main
        step8_main()


if __name__ == "__main__":
    main()
