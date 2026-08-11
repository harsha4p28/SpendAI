import os
import json
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "spend_transactions.parquet")
CSV_INPUT_FILE = os.path.join(DATA_DIR, "spend_transactions.csv")
SUMMARY_OUTPUT = os.path.join(DATA_DIR, "spark_summary.json")

def get_spark_session(app_name="SpendAI-AnomalyDetector"):
    """Initialize local PySpark session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

def run_spend_anomaly_detection():
    print("=" * 60)
    print("  SpendAI - PySpark Distributed Spend Anomaly Engine")
    print("=" * 60)
    
    spark = get_spark_session()
    
    # Read input file
    if os.path.exists(INPUT_FILE):
        df = spark.read.parquet(INPUT_FILE)
        print(f"Ingested parquet spend dataset from: {INPUT_FILE}")
    elif os.path.exists(CSV_INPUT_FILE):
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(CSV_INPUT_FILE)
        print(f"Ingested CSV spend dataset from: {CSV_INPUT_FILE}")
    else:
        raise FileNotFoundError("No input spend dataset found. Run data_engine/generate_data.py first.")

    total_records = df.count()
    print(f"Total Transactions Loaded: {total_records:,}")

    # 1. Statistical Outlier Detection (Z-Score by Category)
    stats_df = df.groupBy("category").agg(
        F.avg("amount").alias("cat_avg_amount"),
        F.stddev("amount").alias("cat_stddev_amount")
    )
    
    df_with_stats = df.join(stats_df, on="category", how="left")
    df_with_zscore = df_with_stats.withColumn(
        "z_score",
        F.when(F.col("cat_stddev_amount") > 0,
               (F.col("amount") - F.col("cat_avg_amount")) / F.col("cat_stddev_amount")
        ).otherwise(0.0)
    )

    # 2. Duplicate Invoice Detection (Spark Window Function)
    # Match same vendor + same amount within 48 hours (172800 seconds)
    df_timestamped = df_with_zscore.withColumn(
        "unix_time", F.unix_timestamp(F.col("timestamp"), "yyyy-MM-dd HH:mm:ss")
    )
    
    window_dup = Window.partitionBy("vendor_name", "amount").orderBy("unix_time")
    
    df_dup = df_timestamped.withColumn(
        "prev_unix_time", F.lag("unix_time", 1).over(window_dup)
    ).withColumn(
        "time_diff_hours", (F.col("unix_time") - F.col("prev_unix_time")) / 3600.0
    ).withColumn(
        "is_duplicate_risk", F.when((F.col("time_diff_hours").isNotNull()) & (F.col("time_diff_hours") <= 48.0), True).otherwise(False)
    )

    # 3. Rule-Based Policy Violations
    df_flagged = df_dup.withColumn(
        "no_po_violation", F.when((F.col("amount") > 5000) & F.col("po_number").isNull(), True).otherwise(False)
    ).withColumn(
        "high_uncategorized_violation", F.when((F.col("amount") > 10000) & (F.col("category") == "Uncategorized"), True).otherwise(False)
    ).withColumn(
        "off_hours_violation", F.when(
            (F.hour(F.col("timestamp")).between(1, 4)) & (F.col("payment_method") == "Employee Expense Reimbursement") & (F.col("amount") > 5000), True
        ).otherwise(False)
    )

    # Combine into overall Risk Score and Anomaly Type
    df_final = df_flagged.withColumn(
        "spark_risk_flag",
        F.when(F.col("is_duplicate_risk"), "DUPLICATE_INVOICE")
         .when(F.col("no_po_violation"), "POLICY_LIMIT_NO_PO")
         .when(F.col("high_uncategorized_violation"), "UNCATEGORIZED_HIGH_VALUE")
         .when(F.col("off_hours_violation"), "OFF_HOURS_HIGH_SPEND")
         .when(F.col("z_score") > 3.0, "STATISTICAL_OUTLIER")
         .otherwise("NORMAL")
    )

    # Calculate Aggregates for Dashboard
    anomaly_counts = df_final.groupBy("spark_risk_flag").count().collect()
    anomaly_summary = {row["spark_risk_flag"]: row["count"] for row in anomaly_counts}

    dept_spend = df_final.groupBy("department").agg(
        F.sum("amount").alias("total_spend"),
        F.avg("amount").alias("avg_spend"),
        F.count("transaction_id").alias("tx_count")
    ).orderBy(F.desc("total_spend")).collect()

    top_vendors = df_final.groupBy("vendor_name").agg(
        F.sum("amount").alias("total_vendor_spend"),
        F.count("transaction_id").alias("vendor_tx_count")
    ).orderBy(F.desc("total_vendor_spend")).limit(10).collect()

    summary_data = {
        "total_transactions": total_records,
        "anomalies": anomaly_summary,
        "flagged_total_count": sum([v for k, v in anomaly_summary.items() if k != "NORMAL"]),
        "department_spend": [{ "department": r["department"], "total_spend": round(r["total_spend"], 2), "avg_spend": round(r["avg_spend"], 2), "tx_count": r["tx_count"] } for r in dept_spend],
        "top_vendors": [{ "vendor": r["vendor_name"], "total_spend": round(r["total_vendor_spend"], 2), "tx_count": r["vendor_tx_count"] } for r in top_vendors]
    }

    # Save summary JSON
    with open(SUMMARY_OUTPUT, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Save sample flagged anomalies as CSV for UI inspection
    flagged_df = df_final.filter(F.col("spark_risk_flag") != "NORMAL").select(
        "transaction_id", "invoice_number", "timestamp", "vendor_name", "department",
        "category", "amount", "po_number", "payment_method", "z_score", "spark_risk_flag"
    )
    flagged_output_csv = os.path.join(DATA_DIR, "spark_detected_anomalies.csv")
    flagged_df.toPandas().to_csv(flagged_output_csv, index=False)

    print(f"\nDetection complete!")
    print(f"Flagged Anomalies Count: {summary_data['flagged_total_count']:,} / {total_records:,}")
    print(f"Summary saved to: {SUMMARY_OUTPUT}")
    print(f"Anomalies sample saved to: {flagged_output_csv}")
    
    spark.stop()
    return summary_data

if __name__ == "__main__":
    run_spend_anomaly_detection()
