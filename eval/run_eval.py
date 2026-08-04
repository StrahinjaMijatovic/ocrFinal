"""
Usage:
  python -m eval.run_eval --prompt-variants v1_basic,v2_few_shot --test-set structuring
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from app.prompts.registry import PromptRegistry
from app.services.ollama_client import OllamaClient
from eval.metrics import summarize

EVAL_DIR = Path(__file__).parent


def load_test_cases(test_set: str) -> list[dict]:
    case_dir = EVAL_DIR / "test_cases" / test_set
    cases = []
    for f in sorted(case_dir.glob("*.json")):
        cases.append(json.loads(f.read_text(encoding="utf-8")))
    return cases


async def eval_structuring_variant(variant: str, cases: list[dict]) -> dict:
    reg = PromptRegistry()
    ollama = OllamaClient()
    results, latencies, expected_types = [], [], []

    for case in cases:
        prompt, temp = reg.format("structuring", variant, ocr_text=case["input"])
        t0 = time.time()
        try:
            result = await ollama.generate(prompt, temperature=temp)
        except Exception:
            result = None
        latencies.append(time.time() - t0)
        results.append(result)
        expected_types.append(case.get("expected_type", "OTHER"))

    await ollama.aclose()
    expected_fields = cases[0].get("expected_fields", []) if cases else []
    return summarize(variant, results, expected_types, expected_fields, latencies)


async def main(variants: list[str], test_set: str, output: str) -> None:
    cases = load_test_cases(test_set)
    if not cases:
        print(f"No test cases found in eval/test_cases/{test_set}/")
        return

    print(f"Running eval on {len(cases)} cases for test set '{test_set}'")
    all_results = {}

    for variant in variants:
        print(f"  Evaluating variant: {variant}...")
        if test_set == "structuring":
            result = await eval_structuring_variant(variant, cases)
        else:
            print(f"  Skipping unsupported test set '{test_set}'")
            continue
        all_results[variant] = result
        print(f"    json_valid={result['json_valid_rate']:.0%} "
              f"field_completeness={result['field_completeness']:.0%} "
              f"doc_type={result['doc_type_accuracy']:.0%} "
              f"latency={result['avg_latency_sec']}s")

    if all_results:
        winner = max(all_results, key=lambda v: all_results[v]["doc_type_accuracy"])
        report = {"test_set": test_set, "variants": all_results, "winner": winner}
        out_path = Path(output) / f"eval_{test_set}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nWinner: {winner}")
        print(f"Report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-variants", default="v1_basic,v2_few_shot")
    parser.add_argument("--test-set", default="structuring")
    parser.add_argument("--output", default="eval/results")
    args = parser.parse_args()
    variants = [v.strip() for v in args.prompt_variants.split(",")]
    asyncio.run(main(variants, args.test_set, args.output))
