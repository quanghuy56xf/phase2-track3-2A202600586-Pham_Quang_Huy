"""Node functions for the LangGraph workflow."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


class ClassificationOutput(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    reasoning: str = Field(default="")


CLASSIFY_PROMPT = """You are a multilingual support ticket router. Follow these steps:

STEP 1 — READ: Parse the user query in any language (English, Vietnamese, etc.).

STEP 2 — IDENTIFY INTENT: Determine the primary topic and what the user wants:
- A how-to / FAQ answer?
- A data lookup (order, tracking, status)?
- A side-effect action (refund, delete, email, cancel)?
- Reporting a system failure (timeout, crash, outage)?
- Or truly no discernible topic (only "help", "fix it", "???")?

STEP 3 — CLASSIFY using priority order (pick the FIRST matching route):
1. risky — Side-effect actions the agent must execute: refunds, deletions, sending emails,
   cancellations, account changes, subscription changes. Works in any language.
2. tool — Information lookups requiring data retrieval: order status, tracking, shipment,
   search, lookup. Must mention lookup/status/track/search OR clear data request.
3. missing_info — LAST RESORT ONLY. Use ONLY when the query has NO discernible topic
   or intent (e.g. "Can you fix it?", "help", "ok", "something is wrong" with zero context).
   Do NOT use missing_info when:
   - The query names a topic (password, order, account, email, payment...) even if short
   - The query is a how-to phrase without question words (e.g. "reset password", "cách đổi mật khẩu")
   - The query is in Vietnamese or any non-English language with clear meaning
4. error — User reports system/technical failures: timeout, crash, service unavailable,
   processing failure, connection error. Not generic "something went wrong" without technical context.
5. simple — General FAQ/how-to answerable without tools or risky actions. Includes short
   phrases and questions about passwords, login, settings, policies, etc.

Examples:
- "How do I reset my password?" → simple
- "cách đổi mật khẩu" → simple (short how-to, clear topic)
- "Cách đổi mật khẩu như thế nào?" → simple
- "Please lookup order status for order 12345" → tool
- "Tra cứu đơn hàng 12345" → tool
- "Can you fix it?" → missing_info (no topic at all)
- "Refund this customer and send email" → risky
- "Timeout failure while processing request" → error

User query: {query}

Return exactly one route. Priority: risky > tool > missing_info > error > simple.
Prefer simple or tool over missing_info whenever a topic can be identified."""


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM."""
    import os

    query = state.get("query", "")
    llm = get_llm()
    method = "function_calling" if os.getenv("OPENAI_BASE_URL") else None
    kwargs = {"method": method} if method else {}
    structured = llm.with_structured_output(ClassificationOutput, **kwargs)
    result: ClassificationOutput = structured.invoke(CLASSIFY_PROMPT.format(query=query))
    route = result.route
    risk_level = "high" if route == "risky" else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                reasoning=result.reasoning,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with transient failure simulation."""
    route = state.get("route", "")
    attempt = state.get("attempt", 0)

    if route == "error" and attempt < 2:
        result = "ERROR: Tool execution failed due to transient timeout"
    else:
        query = state.get("query", "")
        result = f"SUCCESS: Mock tool result for '{query[:60]}'"

    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", "tool executed", success="ERROR" not in result)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate."""
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""

    if "ERROR" in latest.upper():
        evaluation_result = "needs_retry"
        message = "tool result requires retry"
    else:
        evaluation_result = "success"
        message = "tool result satisfactory"

    return {
        "evaluation_result": evaluation_result,
        "events": [make_event("evaluate", "completed", message)],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")

    context_parts = [f"User query: {query}"]
    if tool_results:
        context_parts.append(f"Tool results: {tool_results[-1]}")
    if approval:
        context_parts.append(
            f"Approval status: approved={approval.get('approved')}, "
            f"comment={approval.get('comment', '')}"
        )

    prompt = (
        "You are a helpful support agent. Generate a concise, professional response "
        "grounded in the context below.\n\n"
        + "\n".join(context_parts)
    )

    llm = get_llm()
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "final_answer": str(content),
        "events": [make_event("answer", "completed", "answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    question = (
        f"Could you please provide more details? You said: '{query}' — "
        "what specifically needs to be fixed or addressed?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    action = f"Proposed risky action requiring approval: {query}"
    return {
        "proposed_action": action,
        "events": [make_event("risky_action", "completed", "risky action prepared", action=action)],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step."""
    proposed = state.get("proposed_action", "")

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt({"proposed_action": proposed, "message": "Approve this risky action?"})
        if isinstance(decision, dict):
            approved = bool(decision.get("approved", False))
            comment = str(decision.get("comment", ""))
        else:
            approved = False
            comment = "Rejected via interrupt"
    else:
        approved = True
        comment = "Auto-approved in mock mode"

    return {
        "approval": {"approved": approved, "reviewer": "mock-reviewer", "comment": comment},
        "events": [
            make_event(
                "approval",
                "completed",
                f"approval {'granted' if approved else 'denied'}",
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt."""
    attempt = state.get("attempt", 0) + 1
    error_msg = f"Transient failure on attempt {attempt}"
    return {
        "attempt": attempt,
        "errors": [error_msg],
        "events": [make_event("retry", "completed", f"retry attempt {attempt}")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    answer = (
        f"Unable to complete the request after {attempt} attempt(s) "
        f"(max {max_attempts}). The request has been escalated to the dead letter queue."
    )
    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "completed", "max retries exceeded")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
