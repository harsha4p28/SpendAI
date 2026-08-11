import pytest
import os
import json
from agent_engine.audit_agents import SpendAIModelEngine, AuditResult

@pytest.fixture
def audit_engine():
    # Instantiate with use_llm=False for deterministic, fast offline testing
    return SpendAIModelEngine(use_llm=False)

def test_audit_safe_invoice(audit_engine):
    safe_invoice = {
        "invoice_number": "INV-1001",
        "vendor_name": "Amazon Web Services",
        "amount": 450.0,
        "po_number": "PO-99123",
        "receipt_attached": True,
        "category": "IT Software & Cloud"
    }
    result = audit_engine.audit_invoice(safe_invoice)
    assert isinstance(result, AuditResult)
    assert result.risk_score < 35
    assert result.recommended_action == "APPROVE"

def test_audit_restricted_vendor(audit_engine):
    bad_invoice = {
        "invoice_number": "INV-1002",
        "vendor_name": "Apex Global Vendor LLC",
        "amount": 1200.0,
        "po_number": None,
        "receipt_attached": True,
        "category": "Uncategorized"
    }
    result = audit_engine.audit_invoice(bad_invoice)
    assert result.risk_score >= 50
    assert any("restricted vendor" in v.lower() for v in result.violations)

def test_audit_missing_po_limit(audit_engine):
    large_invoice = {
        "invoice_number": "INV-1003",
        "vendor_name": "Staples",
        "amount": 8500.0,
        "po_number": None,
        "receipt_attached": True,
        "category": "Office Supplies"
    }
    result = audit_engine.audit_invoice(large_invoice)
    assert result.risk_score >= 35
    assert any("purchase order" in v.lower() for v in result.violations)

def test_rule_based_explanation_fallback_when_no_llm(audit_engine):
    invoice = {
        "invoice_number": "INV-1004",
        "vendor_name": "Staples",
        "amount": 6000.0,
        "po_number": None,
        "receipt_attached": True,
        "category": "Office Supplies"
    }
    result = audit_engine.audit_invoice(invoice)
    assert "Rule-based enterprise risk audit applied" in result.explanation
