"""
SpendAI - Real Data Conversion
--------------------------------
Converts the 2013 US Senate Q4 Spending dataset into SpendAI's UNSPSC
fine-tuning format (unspsc_train.json / unspsc_eval_holdout.json).

Source: 2013_US_Senate_Q4_Spending_-_ALL.csv
Columns used: PAYEE, PURPOSE, AMOUNT, CATEGORY

WHY THESE DESIGN CHOICES (read before changing):
1. PERSONNEL COMPENSATION / PERSONNEL BENEFITS are EXCLUDED. Those rows are
   payroll, not procurement spend.
2. Category -> UNSPSC mapping is a manual crosswalk (CATEGORY_MAP below)
   since the source data uses the Legislative Branch's own accounting
   categories, not UNSPSC.
3. RISK LABELS: real rows are only ever labeled LOW or MEDIUM (small
   reversal/correction = MEDIUM, everything else = LOW). We do NOT guess
   which real rows are "risky" — a rule like "amount > 3x category norm"
   is a heuristic with real false-positive risk (a legitimately large
   equipment purchase isn't fraud). Instead, HIGH-risk examples are
   SYNTHESIZED by taking real rows and injecting a known anomaly (missing
   vendor, extreme inflated amount, or duplicate invoice) — the same three
   anomaly types generate_data.py's spend_transactions.csv generator uses.
   This way the model still sees real vendor/purpose vocabulary for the
   HIGH class, but ground truth is certain because we controlled the
   injection, not inferred from a threshold that could be wrong.
4. Held-out eval split is by UNIQUE PURPOSE STRING, not by row, so eval
   tests genuinely unseen phrasing rather than a phrasing that also
   appears (verbatim) in train.
"""

import os
import json
import random
import pandas as pd

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SOURCE_CSV = os.path.join(SCRIPT_DIR, "..", "..", "2013_US_Senate_Q4_Spending_-_ALL.csv")

CATEGORY_MAP = {
    "TRAVEL": ("Travel & Lodging", "90121500"),
    "SUPPLIES AND MATERIALS": ("Office Supplies", "44121700"),
    "RENT, COMMUNICATION, UTILITIES": ("IT Software & Cloud", "43231500"),
    "PRINTING AND REPRODUCTION": ("Marketing & Advertising", "82121500"),
    "EQUIPMENT": ("Hardware & Equipment", "43211900"),
    "OTHER SERVICES": ("Professional Services", "80101500"),
    "FRANKED MAIL": ("Logistics & Shipping", "78102100"),
    "TRANSPORTATION OF THINGS": ("Logistics & Shipping", "78102100"),
}
UNCATEGORIZED_UNSPSC = "99999900"


def load_and_clean(csv_path=SOURCE_CSV):
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)
    df["AMOUNT_CLEAN"] = (
        df["AMOUNT"].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    df = df[df["CATEGORY"].isin(CATEGORY_MAP.keys())].copy().reset_index(drop=True)
    df = df.dropna(subset=["PURPOSE"])
    return df


def assign_base_risk(df):
    """Real rows only ever get LOW or MEDIUM — see module docstring rule 3.
    HIGH is synthesized separately in inject_high_risk_examples()."""
    small_reversal = (df["AMOUNT_CLEAN"] < 0) & (df["AMOUNT_CLEAN"] >= -1000)
    risk = pd.Series("LOW", index=df.index)
    risk[small_reversal] = "MEDIUM"
    df["RISK"] = risk
    return df


def build_example(vendor, amount, purpose, category, unspsc, risk):
    prompt = (
        f"Categorize procurement spend item from Vendor '{vendor}' "
        f"(Amount: ${amount:,.2f}): '{purpose}'"
    )
    completion = json.dumps({
        "category": category,
        "unspsc_code": unspsc,
        "risk_assessment": risk,
    })
    return {
        "instruction": (
            "You are a Business Spend Management AI assistant. Categorize "
            "the procurement transaction into UNSPSC standard code, "
            "category, and risk level."
        ),
        "input": prompt,
        "output": completion,
    }


def build_examples_from_rows(rows):
    examples = []
    for _, row in rows.iterrows():
        vendor = row["PAYEE"] if pd.notna(row["PAYEE"]) else "Unknown/Unlisted Vendor"
        category, unspsc = CATEGORY_MAP[row["CATEGORY"]]
        examples.append(build_example(
            vendor, row["AMOUNT_CLEAN"], str(row["PURPOSE"]).strip(),
            category, unspsc, row["RISK"],
        ))
    return examples


def inject_high_risk_examples(source_rows, n):
    """Synthesize HIGH-risk examples from real rows by injecting a known
    anomaly, mirroring the anomaly types in generate_data.py's
    spend_transactions.csv generator. Ground truth is certain because we
    control the injection."""
    examples = []
    anomaly_types = ["MISSING_VENDOR", "EXTREME_AMOUNT", "DUPLICATE_INVOICE"]
    pool = source_rows.sample(min(n, len(source_rows)), random_state=42).to_dict("records")

    for i in range(n):
        row = pool[i % len(pool)]
        anomaly = random.choice(anomaly_types)
        purpose = str(row["PURPOSE"]).strip()
        category, unspsc = CATEGORY_MAP[row["CATEGORY"]]

        if anomaly == "MISSING_VENDOR":
            vendor = "Unknown/Unlisted Vendor"
            amount = row["AMOUNT_CLEAN"] if row["AMOUNT_CLEAN"] > 0 else abs(row["AMOUNT_CLEAN"])
            purpose = f"{purpose} (no vendor on file, no supporting documentation)"
        elif anomaly == "EXTREME_AMOUNT":
            vendor = row["PAYEE"] if pd.notna(row["PAYEE"]) else "Unknown Vendor"
            base = row["AMOUNT_CLEAN"] if row["AMOUNT_CLEAN"] > 0 else 500.0
            amount = round(base * random.uniform(8, 20), 2)  # implausibly large for this purpose
            purpose = f"{purpose} (amount significantly exceeds typical range for this category)"
        else:  # DUPLICATE_INVOICE
            vendor = row["PAYEE"] if pd.notna(row["PAYEE"]) else "Unknown Vendor"
            amount = row["AMOUNT_CLEAN"] if row["AMOUNT_CLEAN"] > 0 else abs(row["AMOUNT_CLEAN"])
            purpose = f"{purpose} (duplicate submission — same invoice already paid this period)"

        examples.append(build_example(vendor, amount, purpose, category, unspsc, "HIGH"))
    return examples


def generate_real_data_split(max_train=2000, max_eval=300, high_risk_fraction=0.15):
    print("Loading real Senate Q4 2013 spending data...")
    df = load_and_clean()
    print(f"Loaded {len(df)} spend-relevant rows (payroll categories excluded).")

    df = assign_base_risk(df)
    print("Base risk distribution (real rows, LOW/MEDIUM only):")
    print(df["RISK"].value_counts().to_string())

    # Split by UNIQUE PURPOSE TEXT so eval tests genuinely unseen phrasing.
    unique_purposes = df["PURPOSE"].dropna().unique().tolist()
    random.shuffle(unique_purposes)
    split_idx = int(len(unique_purposes) * 0.85)
    train_purposes = set(unique_purposes[:split_idx])
    eval_purposes = set(unique_purposes[split_idx:])

    train_df = df[df["PURPOSE"].isin(train_purposes)]
    eval_df = df[df["PURPOSE"].isin(eval_purposes)]

    n_train_high = int(max_train * high_risk_fraction)
    n_eval_high = int(max_eval * high_risk_fraction)
    n_train_real = max_train - n_train_high
    n_eval_real = max_eval - n_eval_high

    def stratified_sample(subset_df, n):
        if len(subset_df) <= n:
            return subset_df.reset_index(drop=True)
        categories = subset_df["CATEGORY"].unique().tolist()
        per_cat = max(1, n // len(categories))
        pieces = []
        for cat in categories:
            g = subset_df[subset_df["CATEGORY"] == cat]
            take = min(len(g), per_cat)
            pieces.append(g.sample(take, random_state=42))
        sampled = pd.concat(pieces) if pieces else subset_df.iloc[0:0]
        if len(sampled) < n:
            remainder = subset_df.drop(sampled.index)
            extra_n = min(len(remainder), n - len(sampled))
            if extra_n > 0:
                extra = remainder.sample(extra_n, random_state=42)
                sampled = pd.concat([sampled, extra])
        return sampled.sample(frac=1, random_state=42).reset_index(drop=True)

    train_real_sample = stratified_sample(train_df, n_train_real)
    eval_real_sample = stratified_sample(eval_df, n_eval_real)

    train_examples = build_examples_from_rows(train_real_sample)
    eval_examples = build_examples_from_rows(eval_real_sample)

    # Synthesize HIGH-risk examples from DIFFERENT source rows for train vs
    # eval (sampled from train_df / eval_df respectively) so the injected
    # anomaly text isn't literally duplicated across the split either.
    train_examples += inject_high_risk_examples(train_df, n_train_high)
    eval_examples += inject_high_risk_examples(eval_df, n_eval_high)

    random.shuffle(train_examples)
    random.shuffle(eval_examples)

    train_categories = {json.loads(e["output"])["category"] for e in train_examples}
    eval_categories = {json.loads(e["output"])["category"] for e in eval_examples}
    missing = eval_categories - train_categories
    if missing:
        print(f"WARNING: categories {missing} appear in eval but not train.")

    train_path = os.path.join(OUTPUT_DIR, "unspsc_train.json")
    eval_path = os.path.join(OUTPUT_DIR, "unspsc_eval_holdout.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(train_path, "w") as f:
        json.dump(train_examples, f, indent=2)
    with open(eval_path, "w") as f:
        json.dump(eval_examples, f, indent=2)

    print(f"\nSaved {len(train_examples)} train ({n_train_real} real + "
          f"{n_train_high} synthesized-HIGH) / {len(eval_examples)} eval "
          f"({n_eval_real} real + {n_eval_high} synthesized-HIGH) to:\n"
          f"  {train_path}\n  {eval_path}")
    print(f"Train categories: {sorted(train_categories)}")
    print(f"Eval categories: {sorted(eval_categories)}")


if __name__ == "__main__":
    generate_real_data_split()
