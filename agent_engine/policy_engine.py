import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

DEFAULT_POLICY_PATH = os.path.join(os.path.dirname(__file__), "policy_rules.json")

def load_policy(file_path: Optional[str] = None) -> Dict[str, Any]:
    """Load policy rules configuration from JSON file."""
    path = file_path or DEFAULT_POLICY_PATH
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "max_receiptless_amount": 50.0,
        "mandatory_po_threshold": 5000.0,
        "high_risk_uncategorized_threshold": 10000.0,
        "restricted_vendors": ["Apex Global Vendor LLC"],
        "off_hours_restriction_window": {
            "start_hour": 1,
            "end_hour": 4,
            "max_expense_amount": 1000.0
        }
    }

def check_missing_po(amount: float, po_number: Optional[str], policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    """Rule 1: Mandatory Purchase Order (PO) for transactions exceeding policy threshold."""
    mandatory_po = policy.get("mandatory_po_threshold", 5000.0)
    has_po = po_number is not None and str(po_number).strip() != "" and str(po_number).lower() != "none"
    if amount > mandatory_po and not has_po:
        return True, f"Purchase Order (PO) missing for spend exceeding ${mandatory_po:,.2f}", 35
    return False, "", 0

def check_restricted_vendor(vendor_name: str, policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    """Rule 2: Restricted / Blacklisted Vendors check."""
    restricted_vendors = policy.get("restricted_vendors", [])
    vendor_clean = str(vendor_name or "").lower().strip()
    if any(rv.lower() in vendor_clean for rv in restricted_vendors):
        return True, f"Vendor '{vendor_name}' is on company restricted vendor list", 50
    return False, "", 0

def check_missing_receipt(amount: float, receipt_attached: bool, policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    """Rule 3: Receipt Requirement for expense exceeding max receiptless limit."""
    max_no_receipt = policy.get("max_receiptless_amount", 50.0)
    if amount > max_no_receipt and not receipt_attached:
        return True, f"Receipt required for expense exceeding ${max_no_receipt:,.2f}", 20
    return False, "", 0

def check_uncategorized_high_value(amount: float, category: str, policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    """Rule 4: High Value Uncategorized Spend Check."""
    high_val_thresh = policy.get("high_risk_uncategorized_threshold", 10000.0)
    if amount > high_val_thresh and str(category).lower() == "uncategorized":
        return True, f"High value transaction (${high_val_thresh/1000:.0f}k+) tagged as Uncategorized", 25
    return False, "", 0

def check_off_hours_high_spend(timestamp_str: Optional[str], amount: float, payment_method: str, policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    """Rule 5: Off-hours high spend reimbursement check."""
    off_hours_cfg = policy.get("off_hours_restriction_window", {})
    start_h = off_hours_cfg.get("start_hour", 1)
    end_h = off_hours_cfg.get("end_hour", 4)
    max_off_hours = off_hours_cfg.get("max_expense_amount", 1000.0)

    if timestamp_str and payment_method == "Employee Expense Reimbursement" and amount > max_off_hours:
        try:
            dt = datetime.strptime(str(timestamp_str), "%Y-%m-%d %H:%M:%S")
            if start_h <= dt.hour <= end_h:
                return True, f"Off-hours spend during prohibited hours ({start_h}:00-{end_h}:00) exceeding ${max_off_hours:,.2f}", 30
        except Exception:
            pass
    return False, "", 0

def evaluate_invoice_policy(invoice: Dict[str, Any], policy: Optional[Dict[str, Any]] = None) -> Tuple[List[str], int]:
    """Evaluates an invoice dict against all active policy rules."""
    pol = policy or load_policy()
    amount = float(invoice.get("amount", 0.0))
    vendor = invoice.get("vendor_name", "")
    po_number = invoice.get("po_number")
    receipt_attached = bool(invoice.get("receipt_attached", True))
    category = invoice.get("category", "Uncategorized")
    payment_method = invoice.get("payment_method", "Direct Invoice")
    timestamp_str = invoice.get("timestamp")

    rules = [
        check_missing_po(amount, po_number, pol),
        check_restricted_vendor(vendor, pol),
        check_missing_receipt(amount, receipt_attached, pol),
        check_uncategorized_high_value(amount, category, pol),
        check_off_hours_high_spend(timestamp_str, amount, payment_method, pol)
    ]

    violations = []
    total_risk_points = 0
    for triggered, msg, points in rules:
        if triggered:
            violations.append(msg)
            total_risk_points += points

    return violations, total_risk_points
