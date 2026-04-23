# DESIGN.md — Agentic RAG over Indian IT Company Financials

## Project Overview

This project implements a small LLM agent that answers questions over a mixed corpus of:
- **Unstructured**: Annual report text files for Infosys, TCS, and Wipro (FY2024)
- **Structured**: A CSV/SQLite database of key financials for FY2021–FY2024
- **Live**: Web search via Tavily API for current prices, news, and recent developments

---

## Agent Loop — Step by Step

The agent loop is implemented in `agent.py` as a `while` loop with a hard step counter. Here is what happens for every question:

1. **Refusal check** (before any tool call): The question is scanned for investment advice triggers (`invest`, `buy stock`, `portfolio`, etc.). If matched, the agent returns a polite refusal immediately without calling any tool.

2. **Claude API call with tools**: The current conversation history is sent to `claude-haiku-4-5` with the three tool definitions. Claude returns either a final text answer (`stop_reason=end_turn`) or one or more tool calls.

3. **Tool dispatch**: For each tool call Claude requests, the agent calls the appropriate Python function:
   - `search_docs(query)` → FAISS semantic search
   - `query_data(question, sql?)` → SQLite query
   - `web_search(query)` → Tavily API (or mock results)

4. **Step counter increment**: Each tool call increments the step counter. If the counter reaches **8**, the loop exits immediately and returns a structured refusal — never a guess.

5. **Tool results appended**: Results are added to the conversation as `tool_result` messages, allowing Claude to see all prior retrieval in context.

6. **Repeat**: Claude is called again with the enriched context. It either calls more tools or produces a final answer.

7. **Citation collection**: Every tool call appends its source (filename + chunk, SQL used, or URL + date) to the citation list. The final answer includes all citations.

---

## Tool Schemas

### `search_docs`
| Field | Value |
|---|---|
| Purpose | Semantic search over annual report text |
| Input | `query: str` — natural language question |
| Output | Top-3 chunks: `[{text, source, chunk_id, score}]` |
| When to use | Strategy, management commentary, reasons, qualitative analysis |
| When NOT to use | Specific numbers, margins, revenue — use `query_data` |
| Implementation | sentence-transformers (all-MiniLM-L6-v2) + FAISS IndexFlatL2 |

### `query_data`
| Field | Value |
|---|---|
| Purpose | Structured query over financials table |
| Input | `question: str`, optional `sql: str` |
| Output | `{columns, rows, row_count, sql_used}` |
| When to use | Revenue, margin %, EPS, headcount, year-on-year comparisons |
| When NOT to use | Qualitative information, management commentary |
| Implementation | Rule-based NL-to-SQL converter → SQLite |

### `web_search`
| Field | Value |
|---|---|
| Purpose | Live web search for recent/current information |
| Input | `query: str` — short query under 10 words |
| Output | Top-3 results: `[{title, url, content, published_date}]` |
| When to use | Current stock prices, recent news, post-March 2024 events |
| When NOT to use | Historical data already in CSV or annual reports |
| Implementation | Tavily API (mock fallback when key not set) |

---

## Infinite Loop Prevention

Three mechanisms prevent the agent from looping indefinitely:

1. **Hard step counter**: A Python `int` increments on every tool dispatch. A guard at the top of the loop checks `if step >= MAX_STEPS` and returns a `cap_exceeded` status. This is enforced in code — not just in the prompt.

2. **Refusal before loop**: Questions matching investment advice patterns never enter the loop at all, preventing tool calls on unanswerable requests.

3. **`end_turn` exit**: When Claude sets `stop_reason=end_turn`, the loop exits immediately regardless of steps used.

---

## Data Sources

- Annual reports: Represented as `.txt` files in `data/pdfs/`. In a real deployment, these would be extracted from the actual PDFs using `pypdf` or similar.
- Financials: `data/financials.csv` — 12 rows × 7 columns covering 3 companies × 4 years. Loaded into `data/financials.db` at setup time.

---

## Known Limitations

1. **NL-to-SQL is rule-based**: The converter handles common patterns but will fail on complex or ambiguous phrasings. Replacing it with LLM-generated SQL would improve robustness.
2. **No PDF ingestion**: Annual reports are pre-converted text files. A production version would use `pypdf` to extract and chunk PDFs directly.
3. **Mock web search**: Without a `TAVILY_API_KEY`, the web_search tool returns realistic but static mock data.
4. **Single-model**: The agent uses Claude Haiku for cost efficiency. Switching to Claude Sonnet would improve reasoning on complex multi-tool questions.
