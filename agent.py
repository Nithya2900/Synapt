"""
agent.py  —  Agentic RAG over Indian IT Company Financials
LLM: Google Gemini via the NEW google-genai SDK (NOT google-generativeai)

Usage:
    python agent.py "What was Infosys operating margin in FY2024?"
    python agent.py          <- interactive mode
"""

import os, sys, json, time
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ── NEW SDK only — never import google.generativeai ───────────────────────────
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(__file__))
from tools.search_docs import search_docs, format_results as fmt_docs
from tools.query_data  import query_data,  format_result  as fmt_data
from tools.web_search  import web_search,  format_results as fmt_web

# ─── Config ───────────────────────────────────────────────────────────────────
MAX_STEPS = 8
MODEL     = "gemini-2.5-flash"        # free-tier model name for new SDK
REFUSAL_TRIGGERS = [
    "invest","buy stock","sell stock","which stock","should i buy",
    "portfolio","financial advice","trading advice",
]

SYSTEM_INSTRUCTION = (
    "You are a financial research assistant with access to annual reports and "
    "financial data for three Indian IT companies: Infosys, TCS, and Wipro.\n"
    "Tools available:\n"
    "  search_docs  — qualitative/narrative info from annual reports\n"
    "  query_data   — financial figures (revenue, margin, EPS, headcount)\n"
    "  web_search   — recent news and live stock prices\n\n"
    "Pick the most appropriate tool. For questions needing numbers AND explanation, "
    "call query_data first then search_docs. "
    "Stop calling tools once you have enough to compose a clear, cited answer. "
    "If a question cannot be answered from your sources, say so honestly."
)

# ─── Tool schemas ─────────────────────────────────────────────────────────────
TOOL_LIST = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_docs",
        description=(
            "Semantic search over FY2024 annual reports for Infosys, TCS, Wipro. "
            "Use for strategy, management commentary, reasons behind results, risks, "
            "AI platforms, leadership decisions, or any qualitative narrative. "
            "Do NOT use for specific numbers — use query_data for those."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"query": types.Schema(
                type=types.Type.STRING,
                description="Natural language question about qualitative information.")},
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="query_data",
        description=(
            "Query the financials database (Infosys/TCS/Wipro, FY2021-FY2024). "
            "Use for revenue, operating margin, net profit, EPS, headcount, or rankings. "
            "Columns: company, year, revenue_cr, op_margin_pct, net_profit_cr, eps, headcount. "
            "Do NOT use for qualitative information."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "question": types.Schema(type=types.Type.STRING,
                    description="Natural language question about financial figures."),
                "sql": types.Schema(type=types.Type.STRING,
                    description="Optional raw SQL. Table name: financials."),
            },
            required=["question"],
        ),
    ),
    types.FunctionDeclaration(
        name="web_search",
        description=(
            "Search the live web for current stock prices, recent news, analyst ratings, "
            "or any information after March 2024. "
            "Do NOT use for historical data already in the CSV or annual reports."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"query": types.Schema(
                type=types.Type.STRING,
                description="Short search query under 10 words.")},
            required=["query"],
        ),
    ),
])

# ─── Dataclasses ──────────────────────────────────────────────────────────────
@dataclass
class Step:
    step: int; tool: str; input: dict; output: str; latency_ms: float

@dataclass
class AgentTrace:
    question: str
    steps: list        = field(default_factory=list)
    final_answer: str  = ""
    citations: list    = field(default_factory=list)
    status: str        = "success"
    total_ms: float    = 0.0
    timestamp: str     = field(default_factory=lambda: datetime.now().isoformat())

# ─── Agent ────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self):
        key = os.getenv("GEMINI_API_KEY","")
        if not key or "your_gemini" in key:
            raise ValueError(
                "\nGEMINI_API_KEY not set!\n"
                "1. Visit aistudio.google.com  →  Get API Key  →  Create API key\n"
                "2. Open .env (not .env.example) in Notepad\n"
                "3. Add:  GEMINI_API_KEY=AIza...\n"
                "4. Save and rerun.\n"
            )
        self.client = genai.Client(api_key=key)

    # ── tool dispatcher ───────────────────────────────────────────────────────
    def _call_tool(self, name: str, args: dict) -> tuple[str, list[str]]:
        cites = []
        if name == "search_docs":
            res = search_docs(args.get("query",""))
            out = fmt_docs(res)
            cites = [f"search_docs({r['source']}, chunk {r['chunk_id']})" for r in res]
        elif name == "query_data":
            res = query_data(question=args.get("question",""), sql=args.get("sql"))
            out = fmt_data(res)
            cites = [f"query_data(financials.csv, {res.get('sql_used','')})"]
        elif name == "web_search":
            res = web_search(args.get("query",""))
            out = fmt_web(res)
            cites = [f"web_search({r['url']}, {r.get('published_date','')})"
                     for r in res if r.get("url")]
        else:
            out = f"Unknown tool: {name}"
        return out, cites

    # ── main run ──────────────────────────────────────────────────────────────
    def run(self, question: str) -> AgentTrace:
        trace = AgentTrace(question=question)
        t0    = time.time()

        # refusal gate
        if any(t in question.lower() for t in REFUSAL_TRIGGERS):
            trace.final_answer = (
                "I can share factual financial data about Infosys, TCS, and Wipro, "
                "but I cannot give investment advice. "
                "Please consult a SEBI-registered financial advisor."
            )
            trace.status  = "refusal"
            trace.total_ms = (time.time()-t0)*1000
            return trace

        # conversation history as plain dicts (new SDK style)
        history: list[types.Content] = []
        all_cites: list[str] = []
        step = 0

        # seed with user question
        current = [types.Content(role="user",
                                 parts=[types.Part.from_text(text=question)])]

        while step < MAX_STEPS:
            resp = self.client.models.generate_content(
                model=MODEL,
                contents=history + current,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[TOOL_LIST],
                    temperature=0.1,
                ),
            )

            # record what the model said
            history += current
            history.append(types.Content(
                role="model",
                parts=resp.candidates[0].content.parts,
            ))

            # find function calls
            fn_calls = [p.function_call
                        for p in resp.candidates[0].content.parts
                        if p.function_call is not None]

            if not fn_calls:
                trace.final_answer = resp.text
                break

            # execute tools
            result_parts = []
            for fc in fn_calls:
                step += 1
                if step > MAX_STEPS:
                    break
                ts = time.time()
                out, cites = self._call_tool(fc.name, dict(fc.args))
                trace.steps.append(Step(step, fc.name, dict(fc.args),
                                        out, (time.time()-ts)*1000))
                all_cites.extend(cites)
                result_parts.append(types.Part.from_function_response(
                    name=fc.name, response={"result": out}
                ))

            if step >= MAX_STEPS:
                trace.final_answer = (
                    f"[HARD CAP] Agent reached {MAX_STEPS} tool calls. "
                    "Cannot produce a complete answer for this question."
                )
                trace.status = "cap_exceeded"
                break

            current = [types.Content(role="user", parts=result_parts)]

        if not trace.final_answer and trace.status == "success":
            trace.final_answer = "Insufficient information found to answer this question."

        trace.citations = list(dict.fromkeys(all_cites))
        trace.total_ms  = (time.time()-t0)*1000
        return trace

# ─── Display ──────────────────────────────────────────────────────────────────
def print_trace(t: AgentTrace):
    W = 72
    print("="*W)
    print(f"Question : {t.question}")
    print(f"Status   : {t.status.upper()}")
    print(f"Time     : {t.total_ms:.0f} ms")
    print("-"*W)
    for s in t.steps:
        print(f"\nStep {s.step}: {s.tool}  ({s.latency_ms:.0f} ms)")
        print(f"  Input : {json.dumps(dict(s.input))}")
        preview = s.output[:400].replace("\n","\n  ")
        print(f"  Output:\n  {preview}")
        if len(s.output)>400: print(f"  ...({len(s.output)-400} more chars)")
    print(f"\n{'─'*W}")
    print(f"Final Answer:\n{t.final_answer}")
    print(f"\nCitations ({len(t.citations)}):")
    for c in t.citations: print(f"  • {c}")
    print(f"\nSteps used: {len(t.steps)} / {MAX_STEPS}")
    print("="*W)

def save_trace(t: AgentTrace, path: str):
    with open(path,"w") as f:
        json.dump({"question":t.question,"status":t.status,
                   "steps":[{"step":s.step,"tool":s.tool,
                              "input":dict(s.input),"output":s.output[:300],
                              "ms":s.latency_ms} for s in t.steps],
                   "final_answer":t.final_answer,"citations":t.citations},f,indent=2)
    print(f"Saved → {path}")

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = Agent()
    if len(sys.argv) > 1:
        trace = agent.run(" ".join(sys.argv[1:]))
        print_trace(trace)
    else:
        print("Agentic RAG — Indian IT Financials  |  Gemini Free Tier")
        print("Commands: 'quit' to exit | 'save' to save last trace\n")
        last = None
        while True:
            try:    q = input("You: ").strip()
            except: print("\nBye!"); break
            if not q: continue
            if q.lower()=="quit": break
            if q.lower()=="save" and last:
                save_trace(last, f"trace_{int(time.time())}.json"); continue
            last = agent.run(q)
            print_trace(last)
            print()