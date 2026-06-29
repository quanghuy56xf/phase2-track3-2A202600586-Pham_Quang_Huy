"""Hard scenario batch — ad-hoc evaluation (not part of grading)."""

from __future__ import annotations

import json
from pathlib import Path

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.metrics import metric_from_state
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

HARD_CASES = [
    ("H01", "Cancel my subscription immediately", Route.RISKY, True),
    ("H02", "What's my order status?", Route.TOOL, False),
    ("H03", "Something went wrong", Route.MISSING_INFO, False),
    ("H04", "Why did the order lookup fail with timeout?", Route.ERROR, False),
    ("H05", "Update email and refund order 99999", Route.RISKY, True),
    ("H06", "Help", Route.MISSING_INFO, False),
    ("H07", "Hoàn tiền cho khách hàng ngay", Route.RISKY, True),
    ("H08", "Tra cứu trạng thái đơn hàng 12345", Route.TOOL, False),
    ("H09", "Delete all user data permanently", Route.RISKY, True),
    ("H10", "The system crashed and I lost my work", Route.ERROR, False),
    ("H11", "Can you lookup and then refund order 55?", Route.RISKY, False),
    ("H12", "ok", Route.MISSING_INFO, False),
    ("H13", "Send confirmation email to customer", Route.RISKY, False),
    ("H14", "Track shipment ABC123", Route.TOOL, False),
    ("H15", "Fix the timeout error in payment module", Route.ERROR, False),
]


def main() -> None:
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    results = []

    for sid, query, expected, approval_required in HARD_CASES:
        scenario = Scenario(
            id=sid,
            query=query,
            expected_route=expected,
            requires_approval=approval_required,
        )
        state = initial_state(scenario)
        final = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
        metric = metric_from_state(
            final,
            expected.value,
            approval_required,
        )
        results.append(
            {
                "id": sid,
                "query": query,
                "expected": expected.value,
                "actual": final.get("route"),
                "success": metric.success,
                "approval_required": approval_required,
                "approval_observed": metric.approval_observed,
            }
        )
        mark = "PASS" if metric.success else "FAIL"
        q = query[:50].encode("ascii", "replace").decode("ascii")
        print(f"{mark}  {sid}  expected={expected.value:<12} actual={final.get('route'):<12}  {q}")

    passed = sum(1 for r in results if r["success"])
    total = len(results)
    rate = passed / total * 100
    print()
    print(f"RESULT: {passed}/{total} passed ({rate:.1f}%)")

    out = Path("outputs/hard_batch_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": passed, "total": total, "rate": rate, "results": results}, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
