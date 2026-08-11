"""
SpendAI - Pipeline Orchestrator
---------------------------------
Runs the full data -> Spark -> eval pipeline in the correct order so a fresh
clone can go straight to `streamlit run app.py` without manually running each
stage script.

Usage:
    python run_pipeline.py            # skips stages whose output already exists
    python run_pipeline.py --force    # regenerates everything from scratch
"""

import argparse
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SPEND_PARQUET = os.path.join(DATA_DIR, "spend_transactions.parquet")
SUMMARY_JSON = os.path.join(DATA_DIR, "spark_summary.json")
BENCHMARK_JSON = os.path.join(DATA_DIR, "eval_benchmark_results.json")


def stage(title):
    print("\n" + "=" * 70)
    print(f"  STAGE: {title}")
    print("=" * 70)


def run_pipeline(force: bool = False):
    os.makedirs(DATA_DIR, exist_ok=True)

    # Stage 1: synthetic data generation
    stage("1/3 Generating synthetic spend dataset")
    if force or not os.path.exists(SPEND_PARQUET):
        from data_engine.generate_data import generate_spend_data, generate_llm_fine_tuning_dataset
        generate_spend_data(50000)
        generate_llm_fine_tuning_dataset(500)
    else:
        print(f"Skipping — {SPEND_PARQUET} already exists (use --force to regenerate).")

    # Stage 2: PySpark anomaly detection
    stage("2/3 Running PySpark anomaly & duplicate detection")
    if force or not os.path.exists(SUMMARY_JSON):
        from spark_engine.anomaly_detector import run_spend_anomaly_detection
        run_spend_anomaly_detection()
    else:
        print(f"Skipping — {SUMMARY_JSON} already exists (use --force to regenerate).")

    # Stage 3: LLM evaluation benchmark (simulated — see llm_pipeline/eval_model.py)
    stage("3/3 Running (simulated) LLM evaluation benchmark")
    if force or not os.path.exists(BENCHMARK_JSON):
        from llm_pipeline.eval_model import run_evaluation_benchmark
        run_evaluation_benchmark()
    else:
        print(f"Skipping — {BENCHMARK_JSON} already exists (use --force to regenerate).")

    print("\n" + "=" * 70)
    print("  Pipeline complete!")
    print("=" * 70)
    print(f"  - Spend dataset : {SPEND_PARQUET}")
    print(f"  - Spark summary : {SUMMARY_JSON}")
    print(f"  - Benchmark     : {BENCHMARK_JSON}")
    print("\nNext step:\n    streamlit run app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SpendAI data/Spark/eval pipeline end to end.")
    parser.add_argument("--force", action="store_true", help="Regenerate all stages even if output already exists.")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    run_pipeline(force=args.force)
