# Day 08 Lab — LangGraph Agentic Orchestration

Build a production-style LangGraph workflow for a support-ticket agent with state management, conditional routing, retry loops, human-in-the-loop approval, persistence, and metrics.

This lab implements a LangGraph support-ticket agent with LLM classification, conditional routing, retry/HITL, metrics, and a Streamlit demo UI.

---

## Hướng dẫn chạy (Run guide)

### Yêu cầu

- Python **3.11+**
- API key LLM (OpenAI, Anthropic, Gemini, hoặc **DeepSeek**)

### 1. Clone & cài đặt

**Linux / macOS:**

```bash
git clone https://github.com/quanghuy56xf/phase2-track3-2A202600586-Pham_Quang_Huy.git
cd phase2-track3-2A202600586-Pham_Quang_Huy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,openai,ui,sqlite]"
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/quanghuy56xf/phase2-track3-2A202600586-Pham_Quang_Huy.git
cd phase2-track3-2A202600586-Pham_Quang_Huy
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,openai,ui,sqlite]"
```

Hoặc dùng Make:

```bash
make install-all
```

### 2. Cấu hình `.env`

```bash
cp .env.example .env
```

Chỉnh file `.env` — ví dụ với **DeepSeek** (OpenAI-compatible):

```env
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
CHECKPOINTER=memory
LOG_LEVEL=INFO
```

Các provider khác:

| Provider | Biến môi trường |
|---|---|
| OpenAI | `OPENAI_API_KEY=sk-...` |
| Anthropic | `ANTHROPIC_API_KEY=sk-ant-...` |
| Gemini | `GEMINI_API_KEY=AIza...` |
| DeepSeek | `OPENAI_API_KEY` + `OPENAI_BASE_URL=https://api.deepseek.com` |

> DeepSeek cần `method="function_calling"` cho structured output — đã xử lý sẵn trong `classify_node`.

### 3. Chạy test

```bash
make test
# hoặc
python -m pytest
```

Kết quả mong đợi: **25/25 passed** (cần API key hợp lệ trong `.env`).

### 4. Chạy scenarios & chấm điểm local

```bash
make run-scenarios
make grade-local
```

Hoặc không dùng Make:

```bash
python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json
python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json
```

Output:
- `outputs/metrics.json` — kết quả từng scenario
- `reports/lab_report.md` — báo cáo tự sinh

### 5. Demo UI (Streamlit)

```bash
make ui
```

Hoặc:

```bash
python -m streamlit run streamlit_app.py
```

Mở trình duyệt: **http://localhost:8501**

- Chọn scenario mẫu (S01–S07) ở sidebar, hoặc tự nhập câu hỏi
- Bấm **Chạy agent** → xem route, câu trả lời, audit events

> Chạy `streamlit_app.py` ở thư mục gốc — **không** chạy trực tiếp `src/langgraph_agent_lab/ui.py`.

### 6. Test batch câu khó (tùy chọn)

```bash
python scripts/run_hard_batch.py
```

Kết quả lưu tại `outputs/hard_batch_results.json`.

### 7. Slide trình bày

Mở file `docs/lab_slide.html` trong trình duyệt → **F11** full screen → **Space / →** chuyển slide.

### 8. Lint & typecheck (tùy chọn)

```bash
make lint
make typecheck
```

### Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `No LLM API key found` | Kiểm tra `.env`, đảm bảo key đúng |
| `401 Incorrect API key` | Key sai/hết hạn — tạo key mới |
| `ImportError: relative import` (Streamlit) | Dùng `streamlit run streamlit_app.py` |
| `This response_format type is unavailable` (DeepSeek) | Đặt `OPENAI_BASE_URL` — code tự dùng function_calling |
| Test smoke bị skip | Chưa set API key trong `.env` |

---

## Quick start (tóm tắt)

```bash
pip install -e ".[dev,openai,ui]"
cp .env.example .env   # chỉnh API key
make test
make run-scenarios
make grade-local
make ui                # demo Streamlit
```

## How you will be graded

| Category | Points | What we look for |
|---|---:|---|
| Architecture & state schema | 15 | Typed state with correct reducers, student-added fields, lean serializable state |
| Graph construction & wiring | 15 | All nodes registered, edges correct, conditional edges work, graph compiles |
| LLM integration | 15 | classify_node + answer_node use real LLM calls (structured output, grounded generation) |
| Graph behavior | 20 | All scenario routes correct, bounded retry loop, HITL approval path, all routes terminate |
| Persistence & recovery | 10 | Checkpointer wired, thread_id per run, state history or crash-resume evidence |
| Metrics & tests | 15 | `metrics.json` valid, scenario coverage, tests pass, meaningful counts |
| Report & demo | 10 | Architecture explanation, metrics table, failure analysis, improvement ideas |

**Grade bands:**
- **90–100**: Production-quality graph + LLM integration + metrics + report + at least one bonus extension
- **75–89**: Core graph works with LLM, metrics valid, report explains trade-offs
- **60–74**: Graph mostly works but LLM integration, persistence, or report incomplete
- **< 60**: Does not run, hard-codes scenarios, or lacks LLM integration/metrics/report

> **Critical rule**: Do NOT hard-code answers to specific scenario queries. Your graph must route based on **LLM classification and state logic**, not by matching exact scenario IDs. We grade with additional hidden scenarios.

---

## LLM Integration Requirements

This lab requires real LLM API calls in specific nodes:

| Node | Requirement | Pattern |
|---|---|---|
| `classify_node` | **MUST use LLM** | Structured output (`.with_structured_output()`) for intent classification |
| `answer_node` | **MUST use LLM** | Grounded response generation using tool_results/context |
| `evaluate_node` | **SHOULD use LLM** (bonus) | LLM-as-judge to evaluate tool results quality |

A helper is provided in `src/langgraph_agent_lab/llm.py` — it reads your API key from `.env` and returns a LangChain chat model.

```bash
# Install your preferred LLM provider
pip install langchain-openai    # for OpenAI
# OR
pip install langchain-anthropic  # for Anthropic

# Configure .env
cp .env.example .env
# Edit .env and set OPENAI_API_KEY or ANTHROPIC_API_KEY
```

---

## Understanding `scenarios.jsonl`

The file `data/sample/scenarios.jsonl` contains **7 sample scenarios** your graph must handle:

```jsonl
{"id":"S01_simple",      "query":"How do I reset my password?",                          "expected_route":"simple"}
{"id":"S02_tool",        "query":"Please lookup order status for order 12345",            "expected_route":"tool"}
{"id":"S03_missing",     "query":"Can you fix it?",                                      "expected_route":"missing_info"}
{"id":"S04_risky",       "query":"Refund this customer and send confirmation email",      "expected_route":"risky"}
{"id":"S05_error",       "query":"Timeout failure while processing request",              "expected_route":"error"}
{"id":"S06_delete",      "query":"Delete customer account after support verification",    "expected_route":"risky"}
{"id":"S07_dead_letter", "query":"System failure cannot recover after multiple attempts", "expected_route":"error", "max_attempts":1}
```

### What each field means

| Field | Purpose |
|---|---|
| `id` | Unique scenario identifier — used in metrics output |
| `query` | The user's support-ticket text — input to your graph |
| `expected_route` | Which route your `classify_node` should pick: `simple`, `tool`, `missing_info`, `risky`, or `error` |
| `requires_approval` | If `true`, your graph must hit the approval/HITL node before answering |
| `should_retry` | If `true`, scenario simulates transient tool failure requiring retry |
| `max_attempts` | Override retry limit (default 3). S07 sets this to 1, so it exhausts retries immediately → dead letter |
| `tags` | Descriptive labels for your reference |

### How scenarios flow through your code

```
scenarios.jsonl  →  scenarios.py loads them  →  cli.py runs each through your graph
                                              →  metrics.py collects results
                                              →  outputs/metrics.json
```

1. `make run-scenarios` reads `data/sample/scenarios.jsonl`
2. For each scenario, it calls `initial_state(scenario)` → `graph.invoke(state)`
3. After execution, it checks: did `actual_route` match `expected_route`? Did HITL fire when required?
4. Results go to `outputs/metrics.json`

### How to design your classification

Your `classify_node` should use an LLM to classify intent. Design a prompt that routes queries:

| Route | Intent |
|---|---|
| `risky` | Actions with side effects: refunds, deletions, sending emails, cancellations |
| `tool` | Information lookups: order status, tracking, search queries |
| `missing_info` | Vague/incomplete queries lacking actionable context |
| `error` | System failures: timeouts, crashes, service unavailable |
| `simple` | General questions answerable without tools or actions |

**Priority matters**: risky > tool > missing_info > error > simple. Design your LLM prompt to respect this priority.

### Adding your own test scenarios

You can add extra lines to `scenarios.jsonl` to test edge cases:

```jsonl
{"id":"S08_custom","query":"Cancel my subscription immediately","expected_route":"risky","requires_approval":true,"tags":["custom"]}
```

The grading script will also test with scenarios you haven't seen.

---

## Step-by-step workflow

### Phase 1: State + nodes (0–90 min) — worth 30 points

1. **`state.py`** — Review existing fields. Add missing fields as you discover them:
   - `evaluation_result` for retry loop gate
   - `pending_question` for clarification flow
   - `proposed_action` for risky action flow
   - `approval` for HITL decisions

2. **`llm.py`** — Review the helper. Configure `.env` with your API key.

3. **`nodes.py`** — Implement all 10 node functions:
   - `classify_node`: **LLM + structured output** for intent classification
   - `tool_node`: mock tool with error simulation
   - `evaluate_node`: tool result quality check (LLM-as-judge for bonus)
   - `answer_node`: **LLM-generated** grounded response
   - `ask_clarification_node`: generate clarification question
   - `risky_action_node`: prepare action for approval
   - `approval_node`: mock approval with optional interrupt()
   - `retry_or_fallback_node`: increment attempt counter
   - `dead_letter_node`: handle max retry exhaustion
   - `finalize_node`: emit final audit event

### Phase 2: Routing + graph (90–150 min) — worth 35 points

4. **`routing.py`** — Implement all 4 routing functions from scratch
5. **`graph.py`** — Build the complete StateGraph:
   - Import and register all 11 nodes
   - Wire fixed + conditional edges
   - All paths must terminate at finalize → END
6. **Verify**: `make test` and `make run-scenarios`

### Phase 3: Persistence (150–180 min) — worth 10 points

7. **`persistence.py`** — Implement SQLite checkpointer
   - Show evidence: thread_id per run, state history, or crash-resume

### Phase 4: Metrics & report (180–240 min) — worth 25 points

8. **`report.py`** — Implement `render_report()` from metrics data
9. **Run**: `make run-scenarios` → `outputs/metrics.json`
10. **Validate**: `make grade-local`
11. **Report**: Fill `reports/lab_report.md`

### Phase 5: Extensions (240+ min) — push toward 90+

Pick one or more:
- **Parallel fan-out**: Use `Send()` for concurrent tool calls
- **Real HITL**: `LANGGRAPH_INTERRUPT=true` with `interrupt()`
- **Streamlit UI**: Build approval/reject interface
- **Time travel**: `get_state_history()` replay
- **Crash recovery**: SQLite checkpoint survives process kill
- **Graph diagram**: `graph.get_graph().draw_mermaid()`

---

## Make commands

| Command | What it does |
|---|---|
| `make install` | Install project + dev + openai dependencies |
| `make install-all` | Install dev + openai + ui + sqlite |
| `make test` | Run pytest (25 tests) |
| `make lint` | Run ruff linter |
| `make typecheck` | Run mypy type checker |
| `make run-scenarios` | Execute all scenarios → `outputs/metrics.json` |
| `make grade-local` | Validate metrics.json schema |
| `make ui` | Launch Streamlit demo (`streamlit_app.py`) |
| `make clean` | Remove caches and generated files |

---

## Submission checklist

- [ ] All `TODO(student)` sections implemented
- [ ] `.env` configured with LLM API key
- [ ] `classify_node` uses real LLM call with structured output
- [ ] `answer_node` uses real LLM call for grounded responses
- [ ] `make test` passes
- [ ] `make run-scenarios` generates valid `outputs/metrics.json`
- [ ] `make grade-local` passes validation
- [ ] `reports/lab_report.md` completed with architecture, metrics, and analysis
- [ ] Can explain at least one route and one failure mode during demo

**For 90+ points, also include:**
- [ ] At least one bonus extension (persistence, parallel fan-out, HITL, time travel, diagram)
- [ ] Evidence of extension in report (screenshot, log output, or diagram)

---

## Common pitfalls

1. **Missing state fields**: The starter intentionally omits some fields from `AgentState`. You must add `evaluation_result`, `pending_question`, `proposed_action`, and `approval` as you implement nodes that need them.

2. **LLM structured output**: Use `.with_structured_output(YourModel)` to get reliable classification. Raw text parsing is fragile and will fail on hidden test scenarios.

3. **Unbounded retry**: Always check `attempt < max_attempts` in `route_after_retry`. Without this bound, error scenarios loop forever.

4. **Graph wiring**: Every path must end at `finalize → END`. Missing this means the graph hangs for some scenarios.

5. **SqliteSaver API**: In `langgraph-checkpoint-sqlite` 3.x, use `SqliteSaver(conn=sqlite3.connect(...))` not `SqliteSaver.from_conn_string()`.

6. **API key not set**: If you get "No LLM API key found", check your `.env` file and make sure it's loaded (use `python-dotenv` or export manually).
