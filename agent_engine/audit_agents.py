import os
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

POLICY_PATH = os.path.join(os.path.dirname(__file__), "policy_rules.json")

class AuditResult(BaseModel):
    invoice_number: str
    vendor_name: str
    amount: float
    risk_score: int = Field(description="Risk score from 0 (safe) to 100 (high fraud/policy risk)")
    risk_level: str = Field(description="LOW, MEDIUM, HIGH, or CRITICAL")
    violations: List[str] = Field(default_factory=list)
    recommended_action: str = Field(description="APPROVE, FLAG_FOR_HUMAN_REVIEW, or AUTOMATIC_REJECT")
    explanation: str

class SpendAIModelEngine:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.policy = self._load_policy()
        self.llm = self._init_llm()

    def _load_policy(self) -> Dict[str, Any]:
        if os.path.exists(POLICY_PATH):
            with open(POLICY_PATH, "r") as f:
                return json.load(f)
        return {
            "max_receiptless_amount": 50.0,
            "mandatory_po_threshold": 5000.0,
            "restricted_vendors": ["Apex Global Vendor LLC"]
        }

    def _init_llm(self):
        if self.groq_api_key and "gsk_" in self.groq_api_key:
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=self.groq_api_key, temperature=0.1)
            except Exception as e:
                print(f"Notice: Groq API init failed ({e}). Falling back to rule-based agent engine.")
        elif self.openai_api_key and "sk-" in self.openai_api_key:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model_name="gpt-3.5-turbo", api_key=self.openai_api_key, temperature=0.1)
            except Exception as e:
                print(f"Notice: OpenAI API init failed ({e}). Falling back to rule-based agent engine.")
        return None

    def audit_invoice(self, invoice: Dict[str, Any]) -> AuditResult:
        inv_no = invoice.get("invoice_number", "INV-UNKNOWN")
        vendor = invoice.get("vendor_name", "Unknown Vendor")
        amount = float(invoice.get("amount", 0.0))
        po_number = invoice.get("po_number")
        receipt_attached = invoice.get("receipt_attached", True)
        category = invoice.get("category", "Uncategorized")
        payment_method = invoice.get("payment_method", "Direct Invoice")
        
        violations = []
        risk_points = 0
        
        # Rule 1: Mandatory Purchase Order for large amounts
        mandatory_po = self.policy.get("mandatory_po_threshold", 5000.0)
        if amount > mandatory_po and not po_number:
            violations.append(f"Purchase Order (PO) missing for spend exceeding ${mandatory_po:,.2f}")
            risk_points += 35

        # Rule 2: Restricted / Blacklisted Vendors
        restricted_vendors = self.policy.get("restricted_vendors", [])
        if any(rv.lower() in vendor.lower() for rv in restricted_vendors):
            violations.append(f"Vendor '{vendor}' is on company restricted vendor list")
            risk_points += 50

        # Rule 3: Receipt Requirement
        max_no_receipt = self.policy.get("max_receiptless_amount", 50.0)
        if amount > max_no_receipt and not receipt_attached:
            violations.append(f"Receipt required for expense exceeding ${max_no_receipt:,.2f}")
            risk_points += 20

        # Rule 4: High Value Uncategorized Spend
        if amount > 10000.0 and category == "Uncategorized":
            violations.append("High value transaction ($10k+) tagged as Uncategorized")
            risk_points += 25

        # Attempt LLM agent reasoning if LLM is active
        llm_reasoning = ""
        if self.llm:
            try:
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_template(
                    "You are an enterprise spend auditor. Review invoice {inv_no} from Vendor '{vendor}' for ${amount} in category '{category}'. "
                    "Rule Violations identified: {violations}. "
                    "Provide a 2-sentence executive summary of the risk."
                )
                chain = prompt | self.llm
                res = chain.invoke({"inv_no": inv_no, "vendor": vendor, "amount": amount, "category": category, "violations": str(violations)})
                llm_reasoning = res.content
            except Exception:
                llm_reasoning = "Rule-based enterprise risk audit applied."

        final_risk_score = min(risk_points, 100)
        
        if final_risk_score >= 70:
            risk_level = "CRITICAL"
            action = "AUTOMATIC_REJECT"
        elif final_risk_score >= 35:
            risk_level = "HIGH"
            action = "FLAG_FOR_HUMAN_REVIEW"
        elif final_risk_score >= 15:
            risk_level = "MEDIUM"
            action = "FLAG_FOR_HUMAN_REVIEW"
        else:
            risk_level = "LOW"
            action = "APPROVE"

        explanation = llm_reasoning if llm_reasoning else f"Invoice audited against BSM policy rules. {len(violations)} violations flagged."

        return AuditResult(
            invoice_number=inv_no,
            vendor_name=vendor,
            amount=amount,
            risk_score=final_risk_score,
            risk_level=risk_level,
            violations=violations,
            recommended_action=action,
            explanation=explanation
        )

if __name__ == "__main__":
    engine = SpendAIModelEngine()
    test_invoice = {
        "invoice_number": "INV-99012",
        "vendor_name": "Apex Global Vendor LLC",
        "amount": 18500.0,
        "po_number": None,
        "receipt_attached": False,
        "category": "Uncategorized"
    }
    result = engine.audit_invoice(test_invoice)
    print("Test Audit Result:")
    print(json.dumps(result.dict(), indent=2))
