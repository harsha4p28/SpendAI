"""
SpendAI - Shared Policy Evaluation Engine
-------------------------------------------
Single source of truth for spend-policy rule checks, used by BOTH:
  - agent_engine/audit_agents.py   (live, per-invoice audit via SpendAIModelEngine)
  - spark_engine/anomaly_detector.py (batch, distributed anomaly detection)
"""

import os
import json
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

POLICY_PATH = os.path.join(os.path.dirname(__file__), "policy_rules.json")

DEFAULT_POLICY = {
    "max_receiptless_amount": 50.0,
    "mandatory_po_threshold": 5000.0,
    "high_risk_uncategorized_threshold": 10000.0,
    "restricted_vendors": ["Apex Global Vendor LLC"],
    "department_monthly_budget_limits": {},
    "off_hours_restriction_window": {"start_hour": 1, "end_hour": 4, "max_expense_amount": 1000.0},
}


def load_policy(policy_path: str = POLICY_PATH) -> Dict[str, Any]:
    """Load policy_rules.json, falling back to sane defaults for any missing keys."""
    policy = dict(DEFAULT_POLICY)
    if os.path.exists(policy_path):
        with open(policy_path, "r") as f:
            loaded = json.load(f)
        policy.update(loaded)
    return policy


def check_missing_po(amount: float, po_number: Optional[str], policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    threshold = policy.get("mandatory_po_threshold", 5000.0)
    has_po = po_number is not None and str(po_number).strip() != "" and str(po_number).lower() != "none"
    if amount > threshold and not has_po:
        return True, f"Purchase Order (PO) missing for spend exceeding ${threshold:,.2f}", 35
    return False, "", 0


def check_restricted_vendor(vendor: str, policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    restricted_vendors = policy.get("restricted_vendors", [])
    if any(rv.lower() in str(vendor or "").lower() for rv in restricted_vendors):
        return True, f"Vendor '{vendor}' is on company restricted vendor list", 50
    return False, "", 0


def check_missing_receipt(amount: float, receipt_attached: bool, policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    threshold = policy.get("max_receiptless_amount", 50.0)
    if amount > threshold and not receipt_attached:
        return True, f"Receipt required for expense exceeding ${threshold:,.2f}", 20
    return False, "", 0


def check_high_value_uncategorized(amount: float, category: str, policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    threshold = policy.get("high_risk_uncategorized_threshold", 10000.0)
    if amount > threshold and str(category or "").lower() == "uncategorized":
        return True, f"High value transaction (${threshold:,.0f}+) tagged as Uncategorized", 25
    return False, "", 0


def check_off_hours_high_spend(
    hour: int, payment_method: str, amount: float, policy: Dict[str, Any]
) -> Tuple[bool, str, int]:
    window = policy.get("off_hours_restriction_window", DEFAULT_POLICY["off_hours_restriction_window"])
    start_hour = window.get("start_hour", 1)
    end_hour = window.get("end_hour", 4)
    max_amount = window.get("max_expense_amount", 1000.0)
    if (
        start_hour <= hour <= end_hour
        and payment_method == "Employee Expense Reimbursement"
        and amount > max_amount
    ):
        return True, (
            f"Employee expense reimbursement of ${amount:,.2f} submitted between "
            f"{start_hour:02d}:00-{end_hour:02d}:00 exceeds off-hours limit of ${max_amount:,.2f}"
        ), 30
    return False, "", 0


def evaluate_department_budget(
    department: str, department_spend_to_date: float, policy: Dict[str, Any]
) -> Tuple[bool, str, int]:
    """Optional check: flag if a department's cumulative spend exceeds its monthly limit."""
    limits = policy.get("department_monthly_budget_limits", {})
    limit = limits.get(department)
    if limit is not None and department_spend_to_date > limit:
        return True, (
            f"Department '{department}' cumulative spend ${department_spend_to_date:,.2f} "
            f"exceeds monthly budget limit ${limit:,.2f}"
        ), 20
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

    hour = -1
    if timestamp_str:
        try:
            dt = datetime.strptime(str(timestamp_str), "%Y-%m-%d %H:%M:%S")
            hour = dt.hour
        except Exception:
            pass

    rules = [
        check_missing_po(amount, po_number, pol),
        check_restricted_vendor(vendor, pol),
        check_missing_receipt(amount, receipt_attached, pol),
        check_high_value_uncategorized(amount, category, pol),
        check_off_hours_high_spend(hour, payment_method, amount, pol),
    ]

    violations = []
    total_risk_points = 0
    for triggered, msg, points in rules:
        if triggered:
            violations.append(msg)
            total_risk_points += points

    return violations, total_risk_points
