"""
SpendAI - Shared Policy Evaluation Engine
-------------------------------------------
Single source of truth for spend-policy rule checks, used by BOTH:
  - agent_engine/audit_agents.py   (live, per-invoice audit via SpendAIModelEngine)
  - spark_engine/anomaly_detector.py (batch, distributed anomaly detection)

Previously these two engines each hardcoded their own subset of thresholds from
policy_rules.json, so a change to the policy file wouldn't be applied uniformly.
All rule logic and thresholds should live here.
"""

import os
import json
from typing import Any, Dict, Tuple, Optional

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
    if amount > threshold and not po_number:
        return True, f"Purchase Order (PO) missing for spend exceeding ${threshold:,.2f}", 35
    return False, "", 0


def check_restricted_vendor(vendor: str, policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    restricted_vendors = policy.get("restricted_vendors", [])
    if any(rv.lower() in vendor.lower() for rv in restricted_vendors):
        return True, f"Vendor '{vendor}' is on company restricted vendor list", 50
    return False, "", 0


def check_missing_receipt(amount: float, receipt_attached: bool, policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    threshold = policy.get("max_receiptless_amount", 50.0)
    if amount > threshold and not receipt_attached:
        return True, f"Receipt required for expense exceeding ${threshold:,.2f}", 20
    return False, "", 0


def check_high_value_uncategorized(amount: float, category: str, policy: Dict[str, Any]) -> Tuple[bool, str, int]:
    threshold = policy.get("high_risk_uncategorized_threshold", 10000.0)
    if amount > threshold and category == "Uncategorized":
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
