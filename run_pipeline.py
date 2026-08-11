"""
SpendAI - One-Command Pipeline Orchestrator
-------------------------------------------
Runs end-to-end data generation, PySpark anomaly processing, and LLM evaluation benchmarks.

Usage:
    python run_pipeline.py [--force]
"""

import os
import sys
import argparse

from data_engine.generate_data import generate_spend_data, generate_llm_fine_tuning_dataset
from spark_engine.anomaly_detector import run_spend_anomaly_detection
from llm_pipeline.eval_model import run_evaluation_benchmark

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PARQUET_PATH = os.path.join(DATA_DIR, "spend_transactions.parquet")

def main():
    parser = argparse.ArgumentParser(description="SpendAI End-to-End Orchestration Pipeline")
    parser.add_argument("--force", action="store_true", help="Force regeneration of synthetic dataset even if it exists")
    parser.add_argument("--num-records", type=int, default=50000, help="Number of synthetic spend records (default: 50000)")
    parser.add_argument("--num-ft-samples", type=int, default=500, help="Number of fine-tuning dataset samples (default: 500)")
    args = parser.parse_args()

    print("=" * 60)
    print("  SpendAI - End-to-End Orchestration Pipeline")
    print("=" * 60)

    # Stage 1: Data Generation
    if not os.path.exists(PARQUET_PATH) or args.force:
        print("\n[Stage 1/3] Generating synthetic BSM spend dataset & UNSPSC taxonomy samples...")
        generate_spend_data(args.num_records)
        generate_llm_fine_tuning_dataset(args.num_ft_samples)
    else:
        print(f"\n[Stage 1/3] Dataset already exists at {PARQUET_PATH}. (Use --force to regenerate)")

    # Stage 2: PySpark Anomaly Processing
    print("\n[Stage 2/3] Running PySpark Distributed Anomaly & Duplicate Spend Engine...")
    spark_summary = run_spend_anomaly_detection()

    # Stage 3: LLM Evaluation Benchmark
    print("\n[Stage 3/3] Running LLM Model Evaluation Benchmark Suite...")
    eval_summary = run_evaluation_benchmark()

    print("\n" + "=" * 60)
    print("SpendAI Pipeline Execution Completed Successfully!")
    print("=" * 60)
    print(f"Total Transactions Processed : {spark_summary.get('total_transactions', 0):,}")
    print(f"Flagged Spend Anomalies     : {spark_summary.get('flagged_total_count', 0):,}")
    print(f"LLM Accuracy Improvement   : {eval_summary.get('performance_gain', {}).get('accuracy_improvement', 'N/A')}")
    print("\nNext Step: Launch the interactive web dashboard by running:")
    print("   streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
