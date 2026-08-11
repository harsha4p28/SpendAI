import os
import json
import random
from datetime import datetime, timedelta
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEPARTMENTS = ["Engineering", "Sales", "Marketing", "HR", "Operations", "Finance", "Legal"]
CATEGORIES = [
    "IT Software & Cloud", 
    "Office Supplies", 
    "Travel & Lodging", 
    "Professional Services", 
    "Marketing & Advertising", 
    "Hardware & Equipment", 
    "Uncategorized"
]

VENDORS_BY_CATEGORY = {
    "IT Software & Cloud": ["Amazon Web Services", "Microsoft Azure", "Slack Technologies", "GitHub Inc", "Atlassian", "Snowflake Inc"],
    "Office Supplies": ["Staples", "Office Depot", "PaperCorp Solutions", "Herman Miller Furniture"],
    "Travel & Lodging": ["Delta Air Lines", "Marriott International", "Uber for Business", "Airbnb Enterprise"],
    "Professional Services": ["McKinsey & Co", "PwC Consulting", "Deloitte Advisory", "LegalShield LLP"],
    "Marketing & Advertising": ["Google Ads", "Meta Platforms", "LinkedIn Ads", "HubSpot"],
    "Hardware & Equipment": ["Apple Store", "Dell Technologies", "CDW Direct", "Logitech"],
    "Uncategorized": ["Apex Global Vendor LLC", "Misc Merchant 902", "Services Co", "FastPay Transfer"]
}

PAYMENT_METHODS = ["Corporate Card", "Direct Invoice", "Employee Expense Reimbursement"]

def generate_spend_data(num_records=50000):
    print(f"Generating {num_records} synthetic spend records...")
    
    records = []
    base_time = datetime.now() - timedelta(days=90)
    
    for i in range(num_records):
        txn_id = f"TXN-{100000 + i}"
        inv_no = f"INV-{800000 + i}"
        
        # Select category and vendor
        category = random.choices(CATEGORIES, weights=[30, 15, 20, 10, 15, 8, 2])[0]
        vendor = random.choice(VENDORS_BY_CATEGORY[category])
        
        dept = random.choice(DEPARTMENTS)
        emp_id = f"EMP-{random.randint(1001, 1500)}"
        
        # Timestamp
        days_offset = random.randint(0, 90)
        hours_offset = random.randint(0, 23)
        minutes_offset = random.randint(0, 59)
        txn_time = base_time + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
        
        # Amount based on category
        if category == "IT Software & Cloud":
            amount = round(random.uniform(500.0, 45000.0), 2)
        elif category == "Office Supplies":
            amount = round(random.uniform(25.0, 3500.0), 2)
        elif category == "Travel & Lodging":
            amount = round(random.uniform(150.0, 6000.0), 2)
        elif category == "Professional Services":
            amount = round(random.uniform(2000.0, 85000.0), 2)
        elif category == "Marketing & Advertising":
            amount = round(random.uniform(1000.0, 30000.0), 2)
        elif category == "Hardware & Equipment":
            amount = round(random.uniform(300.0, 15000.0), 2)
        else:
            amount = round(random.uniform(100.0, 25000.0), 2)
            
        po_number = f"PO-{random.randint(50000, 99999)}" if (amount > 3000 and random.random() > 0.3) else None
        receipt_attached = random.random() > 0.15
        payment_method = random.choice(PAYMENT_METHODS)
        
        description = f"{category} purchase from {vendor} for {dept} team operations."
        
        records.append({
            "transaction_id": txn_id,
            "invoice_number": inv_no,
            "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            "vendor_name": vendor,
            "employee_id": emp_id,
            "department": dept,
            "category": category,
            "amount": amount,
            "po_number": po_number,
            "receipt_attached": receipt_attached,
            "payment_method": payment_method,
            "description": description,
            "anomaly_flag": "NORMAL"
        })
    
    # Inject Synthetic Anomalies (~3% of data)
    print("Injecting intentional BSM anomalies (Duplicate Invoices, Out-of-Policy Spend, Off-Hours Anomalies)...")
    num_anomalies = int(num_records * 0.03)
    
    for k in range(num_anomalies):
        target_idx = random.randint(0, num_records - 1)
        anomaly_type = random.choice(["DUPLICATE_INVOICE", "POLICY_LIMIT_NO_PO", "UNCATEGORIZED_HIGH_VALUE", "OFF_HOURS_HIGH_SPEND"])
        
        if anomaly_type == "DUPLICATE_INVOICE" and target_idx + 1 < num_records:
            # Create exact duplicate transaction within 2 hours
            original = records[target_idx]
            dup = original.copy()
            dup["transaction_id"] = f"TXN-{200000 + k}"
            dup["invoice_number"] = original["invoice_number"]  # Same invoice number!
            t_orig = datetime.strptime(original["timestamp"], "%Y-%m-%d %H:%M:%S")
            dup["timestamp"] = (t_orig + timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S")
            dup["anomaly_flag"] = "DUPLICATE_INVOICE"
            original["anomaly_flag"] = "DUPLICATE_INVOICE"
            records.append(dup)
            
        elif anomaly_type == "POLICY_LIMIT_NO_PO":
            records[target_idx]["amount"] = round(random.uniform(15000.0, 60000.0), 2)
            records[target_idx]["po_number"] = None  # Policy violation: Spend > $5k without PO
            records[target_idx]["anomaly_flag"] = "POLICY_LIMIT_NO_PO"
            
        elif anomaly_type == "UNCATEGORIZED_HIGH_VALUE":
            records[target_idx]["category"] = "Uncategorized"
            records[target_idx]["amount"] = round(random.uniform(12000.0, 48000.0), 2)
            records[target_idx]["anomaly_flag"] = "UNCATEGORIZED_HIGH_VALUE"
            
        elif anomaly_type == "OFF_HOURS_HIGH_SPEND":
            t_orig = datetime.strptime(records[target_idx]["timestamp"], "%Y-%m-%d %H:%M:%S")
            records[target_idx]["timestamp"] = t_orig.replace(hour=3, minute=15).strftime("%Y-%m-%d %H:%M:%S")
            records[target_idx]["amount"] = round(random.uniform(8000.0, 35000.0), 2)
            records[target_idx]["payment_method"] = "Employee Expense Reimbursement"
            records[target_idx]["anomaly_flag"] = "OFF_HOURS_HIGH_SPEND"

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "spend_transactions.csv")
    parquet_path = os.path.join(OUTPUT_DIR, "spend_transactions.parquet")
    
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df)} records to {csv_path} and {parquet_path}")
    return df

def generate_llm_fine_tuning_dataset(num_samples=500):
    print(f"Generating {num_samples} LLM fine-tuning examples for UNSPSC spend classification...")
    
    unspsc_taxonomy = [
        {"desc_template": "Monthly cloud infrastructure subscription for EC2, S3 bucket storage, and RDS database hosting", "category": "IT Software & Cloud", "unspsc": "43232800", "risk": "LOW"},
        {"desc_template": "Bulk purchase of ergonomic office chairs, dual-monitor desks, and filing cabinets", "category": "Office Supplies", "unspsc": "56110000", "risk": "LOW"},
        {"desc_template": "Roundtrip business flight tickets and 3-night hotel accommodation for annual sales summit", "category": "Travel & Lodging", "unspsc": "90121500", "risk": "LOW"},
        {"desc_template": "Q3 Tax compliance audit, legal risk advisory, and financial bookkeeping retainer fees", "category": "Professional Services", "unspsc": "84110000", "risk": "MEDIUM"},
        {"desc_template": "Targeted digital advertising campaign on Google Search, LinkedIn Ads, and Meta feed", "category": "Marketing & Advertising", "unspsc": "82101800", "risk": "LOW"},
        {"desc_template": "15x Apple MacBook Pro M3 laptops and 4K external monitors for software engineering new hires", "category": "Hardware & Equipment", "unspsc": "43211500", "risk": "MEDIUM"},
        {"desc_template": "Unitemized cash transfer reimbursement to offshore vendor with no itemized receipt attached", "category": "Uncategorized", "unspsc": "99999900", "risk": "HIGH"},
        {"desc_template": "Software license renewal for Enterprise Jira, Confluence, and Slack Pro workspaces", "category": "IT Software & Cloud", "unspsc": "43231500", "risk": "LOW"},
        {"desc_template": "Executive car rental, airport limousine service, and luxury meal expenses", "category": "Travel & Lodging", "unspsc": "78111800", "risk": "MEDIUM"},
        {"desc_template": "Urgent consulting fee payment for external PR agency during product launch crisis", "category": "Professional Services", "unspsc": "80141600", "risk": "HIGH"}
    ]
    
    dataset = []
    for j in range(num_samples):
        item = random.choice(unspsc_taxonomy)
        vendor = random.choice(VENDORS_BY_CATEGORY.get(item["category"], ["Generic Vendor Inc"]))
        amount = round(random.uniform(100.0, 50000.0), 2)
        
        prompt = f"Categorize procurement spend item from Vendor '{vendor}' (Amount: ${amount:,.2f}): '{item['desc_template']}'"
        completion = json.dumps({
            "category": item["category"],
            "unspsc_code": item["unspsc"],
            "risk_assessment": item["risk"]
        })
        
        dataset.append({
            "instruction": "You are a Business Spend Management AI assistant. Categorize the procurement transaction into UNSPSC standard code, category, and risk level.",
            "input": prompt,
            "output": completion
        })
        
    ft_path = os.path.join(OUTPUT_DIR, "unspsc_fine_tuning_dataset.json")
    with open(ft_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved fine-tuning dataset to {ft_path}")

    # Held-out split for real evaluation. Splitting AFTER generation on shuffled
    # indices, rather than filtering by template, is a deliberate compromise: with
    # only 10 hand-written templates, a template-level split would leave whole
    # categories out of either train or eval. This split at least prevents the
    # exact same (vendor, amount) pairing from appearing in both files, and keeps
    # the harness honest about what "held-out" means here vs. a production dataset
    # with far more source templates.
    shuffled = dataset[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.85)
    train_split, eval_split = shuffled[:split_idx], shuffled[split_idx:]

    train_path = os.path.join(OUTPUT_DIR, "unspsc_train.json")
    eval_path = os.path.join(OUTPUT_DIR, "unspsc_eval_holdout.json")
    with open(train_path, "w") as f:
        json.dump(train_split, f, indent=2)
    with open(eval_path, "w") as f:
        json.dump(eval_split, f, indent=2)
    print(f"Saved {len(train_split)} train / {len(eval_split)} held-out eval examples to "
          f"{train_path} / {eval_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate SpendAI synthetic spend & fine-tuning datasets.")
    parser.add_argument("--num-records", type=int, default=50000, help="Number of synthetic spend transactions to generate.")
    parser.add_argument("--num-ft-samples", type=int, default=500, help="Number of LLM fine-tuning examples to generate.")
    args = parser.parse_args()

    generate_spend_data(args.num_records)
    generate_llm_fine_tuning_dataset(args.num_ft_samples)