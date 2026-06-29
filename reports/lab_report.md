# Day 08 Lab Report

## 1. Team / student

- Name: Pham Quang Huy
- Repo/commit: phase2-track3-2A202600586-Pham_Quang_Huy
- Date: 2026-06-29

## 2. Architecture

The graph follows a support-ticket workflow:

`START → intake → classify → [conditional routing] → ... → finalize → END`

- **intake**: normalizes the query
- **classify**: LLM structured output routes to simple/tool/missing_info/risky/error
- **tool → evaluate**: mock tool with retry loop for transient errors
- **risky_action → approval**: HITL gate before executing risky actions
- **retry → dead_letter**: bounded retry with escalation when max_attempts exceeded
- **answer / clarify / dead_letter → finalize**: all paths terminate cleanly

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| messages | append | audit conversation trail |
| tool_results | append | accumulate tool outputs across retries |
| errors | append | track transient failures |
| events | append | append-only audit log for grading |
| route | overwrite | current classified route |
| attempt | overwrite | retry counter |
| evaluation_result | overwrite | retry-loop gate |
| approval | overwrite | latest HITL decision |

## 4. Scenario results

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100% |
| Avg nodes visited | 6.4 |
| Total retries | 3 |
| Total interrupts | 2 |

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | Yes | 0 | 0 |
| S02_tool | tool | tool | Yes | 0 | 0 |
| S03_missing | missing_info | missing_info | Yes | 0 | 0 |
| S04_risky | risky | risky | Yes | 0 | 1 |
| S05_error | error | error | Yes | 2 | 0 |
| S06_delete | risky | risky | Yes | 0 | 1 |
| S07_dead_letter | error | error | Yes | 1 | 0 |

## 5. Failure analysis

1. **Retry / tool failure**: Error-route scenarios simulate transient tool failures. The evaluate node detects `ERROR` in tool results and routes to retry. Without a bounded `max_attempts` check, the graph would loop forever.
2. **Risky action without approval**: Risky routes pass through approval_node before tool execution. Mock mode auto-approves; production would use `LANGGRAPH_INTERRUPT=true` for real HITL.

## 6. Persistence / recovery evidence

MemorySaver checkpointer is wired with a unique `thread_id` per scenario run. SQLite checkpointer is implemented in `persistence.py` for crash-recovery extension (`CHECKPOINTER=sqlite`).

## 7. Extension work

- SQLite checkpointer with WAL mode
- Structured LLM classification via `.with_structured_output()`
- Mock HITL approval with optional real interrupt support

## 8. Improvement plan

Productionize real HITL with a Streamlit approval UI, add LLM-as-judge in evaluate_node, and enable crash-resume testing with SQLite persistence.
