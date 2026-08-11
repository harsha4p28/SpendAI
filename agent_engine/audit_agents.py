import os
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent_engine.policy_engine import load_policy, evaluate_invoice_policy

load_dotenv()
logger = logging.getLogger(__name__)

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
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.policy = load_policy()
        self.llm = self._init_llm() if self.use_llm else None
        self.prompt = self._init_prompt() if self.llm else None

    def _init_llm(self):
        if not self.use_llm:
            return None
        if self.groq_api_key and "gsk_" in self.groq_api_key:
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=self.groq_api_key, temperature=0.1)
            except Exception as e:
                logger.warning("Groq API init failed: %s", e, exc_info=True)
        elif self.openai_api_key and "sk-" in self.openai_api_key:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model_name="gpt-3.5-turbo", api_key=self.openai_api_key, temperature=0.1)
            except Exception as e:
                logger.warning("OpenAI API init failed: %s", e, exc_info=True)
        return None

    def _init_prompt(self):
        try:
            from langchain_core.prompts import ChatPromptTemplate
            return ChatPromptTemplate.from_template(
                "You are an enterprise spend auditor. Review invoice {inv_no} from Vendor '{vendor}' for ${amount} in category '{category}'. "
                "Rule Violations identified: {violations}. "
                "Provide a 2-sentence executive summary of the risk."
            )
        except Exception as e:
            logger.warning("Failed to build ChatPromptTemplate: %s", e, exc_info=True)
            return None

    def audit_invoice(self, invoice: Dict[str, Any]) -> AuditResult:
        inv_no = invoice.get("invoice_number", "INV-UNKNOWN")
        vendor = invoice.get("vendor_name", "Unknown Vendor")
        amount = float(invoice.get("amount", 0.0))
        category = invoice.get("category", "Uncategorized")
        
        # Pure policy evaluation via policy_engine
        violations, risk_points = evaluate_invoice_policy(invoice, self.policy)
        
        # LLM reasoning attempt
        llm_reasoning = ""
        if self.llm and self.prompt:
            try:
                chain = self.prompt | self.llm
                res = chain.invoke({
                    "inv_no": inv_no,
                    "vendor": vendor,
                    "amount": amount,
                    "category": category,
                    "violations": str(violations)
                })
                llm_reasoning = res.content
            except Exception as e:
                logger.warning("LLM audit invocation failed, falling back to rule-based summary: %s", e, exc_info=True)
                llm_reasoning = ""

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

        explanation = llm_reasoning if llm_reasoning else f"Rule-based enterprise risk audit applied. {len(violations)} policy violations flagged."

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
    engine = SpendAIModelEngine(use_llm=True)
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
    print(json.dumps(result.model_dump(), indent=2))
