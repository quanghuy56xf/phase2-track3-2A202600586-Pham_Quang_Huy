from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

queries = [
    ("EN", "How do I reset my password?"),
    ("VI", "Cách đổi mật khẩu như thế nào?"),
    ("VI-short", "cách đổi mật khẩu"),
]

graph = build_graph(checkpointer=build_checkpointer("memory"))
for label, q in queries:
    scenario = Scenario(id=label, query=q, expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    ans = str(result.get("final_answer", ""))[:100]
    print(f"{label:10} route={result['route']:12} nodes={len(result.get('events',[]))}")
    print(f"           query: {q}")
    print(f"           answer: {ans}...")
    print()
