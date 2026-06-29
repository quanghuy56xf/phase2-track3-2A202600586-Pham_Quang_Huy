"""Simple Streamlit UI to test the LangGraph support agent."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import streamlit as st

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import Route, Scenario, initial_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_PATH = PROJECT_ROOT / "data" / "sample" / "scenarios.jsonl"


@st.cache_resource
def get_graph():
    return build_graph(checkpointer=build_checkpointer("memory"))


def main() -> None:
    st.set_page_config(page_title="LangGraph Agent Lab", page_icon="🎫", layout="wide")
    st.title("Day 08 — Support Ticket Agent")
    st.caption("Nhập câu hỏi hoặc chọn scenario mẫu để xem graph route và trả lời.")

    scenarios = load_scenarios(str(SCENARIOS_PATH))
    preset_labels = ["(Tự nhập)"] + [f"{s.id}: {s.query[:50]}..." for s in scenarios]

    with st.sidebar:
        st.header("Scenario mẫu")
        choice = st.selectbox("Chọn preset", preset_labels, index=0)
        preset_idx = preset_labels.index(choice) - 1
        default_query = scenarios[preset_idx].query if preset_idx >= 0 else ""

    query = st.text_area(
        "Câu hỏi support ticket",
        value=default_query,
        height=100,
        placeholder="Ví dụ: How do I reset my password?",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run = st.button("Chạy agent", type="primary", use_container_width=True)

    if not run:
        st.info("Nhập câu hỏi rồi bấm **Chạy agent**.")
        return

    if not query.strip():
        st.error("Vui lòng nhập câu hỏi.")
        return

    with st.spinner("Đang chạy graph (gọi LLM)..."):
        thread_id = f"ui-{uuid.uuid4().hex[:8]}"
        scenario = Scenario(id="ui-test", query=query.strip(), expected_route=Route.SIMPLE)
        state = initial_state(scenario)
        state["thread_id"] = thread_id
        graph = get_graph()
        result = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})

    st.success("Hoàn thành")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Route", result.get("route", "-"))
    m2.metric("Risk", result.get("risk_level", "-"))
    m3.metric("Retries", result.get("attempt", 0))
    m4.metric("Nodes", len(result.get("events", [])))

    st.subheader("Trả lời")
    if result.get("final_answer"):
        st.write(result["final_answer"])
    elif result.get("pending_question"):
        st.write(result["pending_question"])
    else:
        st.warning("Không có final_answer.")

    if result.get("approval"):
        st.subheader("Approval (HITL)")
        st.json(result["approval"])

    if result.get("tool_results"):
        st.subheader("Tool results")
        for item in result["tool_results"]:
            st.code(item)

    if result.get("errors"):
        st.subheader("Errors")
        for err in result["errors"]:
            st.error(err)

    with st.expander("Audit events"):
        st.dataframe(result.get("events", []), use_container_width=True)

    with st.expander("Full state (JSON)"):
        st.code(json.dumps(result, indent=2, ensure_ascii=False, default=str), language="json")


if __name__ == "__main__":
    main()
