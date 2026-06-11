"""Run ablation sweep A0-A5 and produce final comparison table.

Ablation configurations:
  A0: Base OpenOneRec-1.7B (no SFT, no enrichment)
  A1: SFT v1 (heuristic cards, rule-based planner)
  A2: SFT v2 (TextGrad-optimized LLM cards)
  A3: SFT v2 + DPO (RecPO with rollout-beam + counterfactual)
  A4: A3 without enrichment at test time (tests robustness)
  A5: A3 with enrichment at test time (full pipeline)

Gate E: A5 avg R@10 >= A0 + 1.5pp

Usage:
    python scripts/09_run_ablations.py --config configs/ablation_config.yaml
    python scripts/09_run_ablations.py --quick  # 100 samples per task
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ablations")

# FIX: A3/A4/A5 now point at the DPO r16 output (./checkpoints/arec2-dpo-r16/final)
# produced by configs/dpo_config.yaml. The old paths (arec2-dpo-r8/final) did not
# exist after a r16 DPO run, so these ablations were silently SKIPPED.
DPO_ADAPTER = "./checkpoints/arec2-dpo-r16/final"
SFT_V2_ADAPTER = "./checkpoints/arec2-lora-r16-v2/final"
SFT_V1_ADAPTER = "./checkpoints/arec2-lora-r16/final"

# Default ablation configurations
ABLATIONS = {
    "A0": {
        "description": "Base OpenOneRec-1.7B (no SFT, no enrichment)",
        "model": "OpenOneRec/OneRec-1.7B",
        "adapter": None,
        "enrich": False,
    },
    "A1": {
        "description": "SFT v1 (heuristic cards, rule-based planner)",
        "model": "OpenOneRec/OneRec-1.7B",
        "adapter": SFT_V1_ADAPTER,
        "enrich": True,
    },
    "A2": {
        "description": "SFT v2 (TextGrad-optimized LLM cards)",
        "model": "OpenOneRec/OneRec-1.7B",
        "adapter": SFT_V2_ADAPTER,
        "enrich": True,
    },
    "A3": {
        "description": "SFT v2 + DPO (full RecPO)",
        "model": "OpenOneRec/OneRec-1.7B",
        "adapter": DPO_ADAPTER,
        "enrich": True,
    },
    "A4": {
        "description": "A3 without test-time enrichment (robustness test)",
        "model": "OpenOneRec/OneRec-1.7B",
        "adapter": DPO_ADAPTER,
        "enrich": False,
    },
    "A5": {
        "description": "A3 with test-time enrichment (full pipeline)",
        "model": "OpenOneRec/OneRec-1.7B",
        "adapter": DPO_ADAPTER,
        "enrich": True,
    },
}


def run_single_eval(
    ablation_name: str,
    ablation_config: dict,
    eval_config: str,
    tasks: list[str] | None,
    max_samples: int | None,
    batch_size: int,
    output_dir: Path,
) -> dict | None:
    """Run a single ablation evaluation."""
    output_file = output_dir / f"{ablation_name}.json"

    cmd = [
        sys.executable, "scripts/08_eval_recif.py",
        "--model", ablation_config["model"],
        "--enrich", "true" if ablation_config["enrich"] else "false",
        "--config", eval_config,
        "--output", str(output_file),
    ]

    if ablation_config.get("adapter"):
        adapter_path = ablation_config["adapter"]
        if Path(adapter_path).exists():
            cmd.extend(["--adapter", adapter_path])
        else:
            logger.warning("Adapter not found for %s: %s (skipping)", ablation_name, adapter_path)
            return None
    else:
        # A0 (base model, no adapter): pass empty adapter to override the
        # eval script's default DPO adapter path.
        cmd.extend(["--adapter", ""])

    if tasks:
        cmd.extend(["--tasks"] + tasks)
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    if batch_size > 1:
        cmd.extend(["--batch-size", str(batch_size)])

    logger.info("Running %s: %s", ablation_name, ablation_config["description"])
    logger.info("  Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per ablation
        )
        if result.returncode != 0:
            logger.error("  %s FAILED:\n%s", ablation_name, result.stderr[-500:])
            return None

        # Load results
        if output_file.exists():
            with open(output_file, "r") as f:
                return json.load(f)
        return None

    except subprocess.TimeoutExpired:
        logger.error("  %s TIMEOUT (>2h)", ablation_name)
        return None
    except Exception as e:
        logger.error("  %s ERROR: %s", ablation_name, e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Run ablation sweep A0-A5")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                        help="Training config (for store paths)")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Specific tasks (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per task")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 100 samples per task")
    parser.add_argument("--ablations", type=str, nargs="*", default=None,
                        help="Specific ablations to run (e.g., A0 A3 A5)")
    parser.add_argument("--output-dir", type=str, default="results/ablations")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    args = parser.parse_args()

    if args.quick:
        args.max_samples = 100

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which ablations to run
    ablation_names = args.ablations or list(ABLATIONS.keys())

    logger.info("=" * 70)
    logger.info("AREC2 Ablation Sweep")
    logger.info("  Ablations: %s", ablation_names)
    logger.info("  Tasks: %s", args.tasks or "all")
    logger.info("  Max samples/task: %s", args.max_samples or "unlimited")
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 70)

    t_start = time.time()
    all_results = {}

    for name in ablation_names:
        if name not in ABLATIONS:
            logger.warning("Unknown ablation: %s, skipping", name)
            continue

        result = run_single_eval(
            name, ABLATIONS[name], args.config,
            args.tasks, args.max_samples, args.batch_size, output_dir,
        )
        if result:
            all_results[name] = result

    # Print comparison table
    elapsed = time.time() - t_start
    print("\n")
    print("=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    print(f"{'Ablation':<6} {'Description':<50} {'Avg Recall@32':<14} {'Enrich'}")
    print("-" * 80)

    a0_rc32 = None
    for name in ablation_names:
        if name not in all_results:
            print(f"{name:<6} {'[SKIPPED/FAILED]':<50}")
            continue

        result = all_results[name]
        avg_rc32 = result.get("avg_Recall32")
        desc = ABLATIONS[name]["description"][:48]
        enrich_str = "Yes" if ABLATIONS[name]["enrich"] else "No"

        rc32_str = f"{avg_rc32:.4f}" if avg_rc32 is not None else "N/A"
        print(f"{name:<6} {desc:<50} {rc32_str:<14} {enrich_str}")

        if name == "A0" and avg_rc32 is not None:
            a0_rc32 = avg_rc32

    print("-" * 80)

    # Gate E check
    if a0_rc32 is not None and "A5" in all_results:
        a5_rc32 = all_results["A5"].get("avg_Recall32")
        if a5_rc32 is not None:
            improvement = (a5_rc32 - a0_rc32) * 100
            gate_pass = improvement >= 1.5
            status = "PASSED" if gate_pass else "FAILED"
            print(f"\nGate E: A5 - A0 = {improvement:+.2f}pp (threshold: +1.5pp) [{status}]")

    # A4 vs A0 robustness check
    if a0_rc32 is not None and "A4" in all_results:
        a4_rc32 = all_results["A4"].get("avg_Recall32")
        if a4_rc32 is not None:
            diff = (a4_rc32 - a0_rc32) * 100
            print(f"Robustness: A4 - A0 = {diff:+.2f}pp (DPO without enrichment vs baseline)")

    print(f"\nTotal sweep time: {elapsed / 60:.1f} minutes")
    print("=" * 80)

    # Save combined results
    combined_path = output_dir / "ablation_summary.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({
            "ablations": all_results,
            "elapsed_seconds": elapsed,
            "gate_e": {
                "a0_recall32": a0_rc32,
                "a5_recall32": all_results.get("A5", {}).get("avg_Recall32"),
            } if a0_rc32 else None,
        }, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved to %s", combined_path)


if __name__ == "__main__":
    main()