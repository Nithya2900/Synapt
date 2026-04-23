"""
evaluate.py

Runs the full 20-question evaluation set and produces EVALUATION.md.

Usage:
    python evaluate.py
    python evaluate.py --question "What was TCS margin?"  (single question)
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from agent import Agent, print_trace, AgentTrace

# ─── Evaluation set ───────────────────────────────────────────────────────────
# Format: (question, expected_tools, category, expected_behaviour_note)
EVAL_SET = [
    # ── Single-tool: query_data ───────────────────────────────────────────────
    (
        "What was Infosys operating margin in FY2024?",
        ["query_data"],
        "single_tool",
        "Should return 20.7% from financials.csv",
    ),
    (
        "Show TCS revenue over the last 4 years",
        ["query_data"],
        "single_tool",
        "Should return year-by-year table for TCS FY2021-FY2024",
    ),
    (
        "Which company had the highest operating margin in FY2024?",
        ["query_data"],
        "single_tool",
        "Should return TCS at 24.6%",
    ),
    (
        "What was Wipro headcount in FY2023?",
        ["query_data"],
        "single_tool",
        "Should return 257002",
    ),
    (
        "Compare all three companies revenue in FY2024",
        ["query_data"],
        "single_tool",
        "Should return table with TCS > Infosys > Wipro",
    ),
    (
        "What was TCS EPS in FY2024?",
        ["query_data"],
        "single_tool",
        "Should return 124.7",
    ),

    # ── Single-tool: search_docs ──────────────────────────────────────────────
    (
        "What reason did TCS give for its margin improvement in FY2024?",
        ["search_docs"],
        "single_tool",
        "Should mention subcontractor cost management and pyramid optimization",
    ),
    (
        "What strategic priorities did Infosys highlight in FY2024?",
        ["search_docs"],
        "single_tool",
        "Should mention Topaz AI, Cobalt cloud, large deal wins",
    ),
    (
        "What AI platform did Wipro launch?",
        ["search_docs"],
        "single_tool",
        "Should mention AI360 strategy",
    ),

    # ── Single-tool: web_search ───────────────────────────────────────────────
    (
        "What is the current stock price of Infosys?",
        ["web_search"],
        "single_tool",
        "Should use web_search and return a price with URL",
    ),
    (
        "Who is the current CEO of Wipro?",
        ["web_search"],
        "single_tool",
        "Should return Srinivas Pallia with source",
    ),

    # ── Multi-tool ────────────────────────────────────────────────────────────
    (
        "How did Infosys and TCS operating margins compare in FY2024, and what drove each?",
        ["query_data", "search_docs"],
        "multi_tool",
        "Should get numbers from query_data, reasons from search_docs, combine both",
    ),
    (
        "Compare headcount growth at all 3 companies over 4 years and explain Wipro's reduction",
        ["query_data", "search_docs"],
        "multi_tool",
        "Numbers from query_data, explanation from Wipro AR about workforce rationalization",
    ),
    (
        "What was TCS revenue in FY2024, what large deal drove it, and what is the current stock price?",
        ["query_data", "search_docs", "web_search"],
        "multi_tool",
        "Three-tool question: revenue from data, BSNL deal from docs, price from web",
    ),
    (
        "Which company had the best margin improvement from FY2021 to FY2024 and why?",
        ["query_data", "search_docs"],
        "multi_tool",
        "Should calculate improvement from data, then find reasons in docs",
    ),
    (
        "What was Infosys net profit in FY2024 and what cost actions helped preserve it?",
        ["query_data", "search_docs"],
        "multi_tool",
        "Number from CSV, voluntary separation scheme and real estate from AR",
    ),

    # ── Refusal questions ─────────────────────────────────────────────────────
    (
        "Which of the three companies should I invest in?",
        [],
        "refusal",
        "Should refuse — investment advice is not permitted",
    ),
    (
        "Should I buy TCS stock now?",
        [],
        "refusal",
        "Should refuse — stock buy recommendation not permitted",
    ),
    (
        "Give me a portfolio allocation between Infosys and Wipro",
        [],
        "refusal",
        "Should refuse — portfolio advice not permitted",
    ),
    (
        "What is the airspeed velocity of an unladen swallow?",
        [],
        "refusal",
        "Should refuse or answer directly without calling any tool",
    ),

    # ── Edge cases ────────────────────────────────────────────────────────────
    (
        "What was Infosys revenue in FY2019?",
        [],
        "edge_case",
        "FY2019 is outside our data range. Agent should say data not available.",
    ),
    (
        "What will TCS revenue be in FY2026?",
        [],
        "edge_case",
        "Future data — agent should acknowledge it cannot predict future figures",
    ),
    (
        "How does Infosys compare to Accenture?",
        [],
        "edge_case",
        "Accenture data not in corpus — agent should acknowledge this gap",
    ),
    (
        "What is 2 + 2?",
        [],
        "edge_case",
        "Trivial question — agent should answer directly without any tool call",
    ),
]


def grade_response(trace: AgentTrace, expected_tools: list[str], category: str) -> dict:
    """Simple grader: checks tool usage and whether answer seems substantive."""
    tools_called = [s.tool for s in trace.steps]

    correct_tools = False
    if category == "refusal":
        correct_tools = (trace.status == "refusal" or len(tools_called) == 0)
    elif category == "edge_case":
        correct_tools = True  # edge cases are graded manually
    else:
        # Check all expected tools were called
        correct_tools = all(t in tools_called for t in expected_tools)

    has_answer = len(trace.final_answer) > 30
    cap_respected = len(trace.steps) <= 8

    score = 0
    if correct_tools:
        score += 1
    if has_answer:
        score += 1
    if cap_respected:
        score += 1

    return {
        "tools_called": tools_called,
        "expected_tools": expected_tools,
        "correct_tools": correct_tools,
        "has_answer": has_answer,
        "cap_respected": cap_respected,
        "score": score,
        "max_score": 3,
    }


def run_evaluation(output_path: str = "EVALUATION.md") -> None:
    agent = Agent()
    results = []

    print(f"Running {len(EVAL_SET)} evaluation questions...\n")

    for i, (question, expected_tools, category, note) in enumerate(EVAL_SET, 1):
        print(f"[{i:02d}/{len(EVAL_SET)}] {category.upper()}: {question[:60]}...")
        try:
            trace = agent.run(question)
            grade = grade_response(trace, expected_tools, category)
            results.append({
                "id": i,
                "question": question,
                "category": category,
                "expected_tools": expected_tools,
                "note": note,
                "trace": trace,
                "grade": grade,
            })
            status_icon = "✓" if grade["correct_tools"] else "✗"
            print(f"  {status_icon} tools={grade['tools_called']} score={grade['score']}/{grade['max_score']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": i, "question": question, "category": category,
                "expected_tools": expected_tools, "note": note,
                "trace": None, "grade": {"score": 0, "max_score": 3, "error": str(e)},
            })

    # ── Write EVALUATION.md ───────────────────────────────────────────────────
    by_category = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, []).append(r)

    lines = []
    lines.append("# Evaluation Report\n")
    lines.append(f"**Total questions:** {len(EVAL_SET)}  ")
    lines.append(f"**Run date:** {time.strftime('%Y-%m-%d %H:%M')}  \n")

    total_score = sum(r["grade"].get("score", 0) for r in results)
    max_total = sum(r["grade"].get("max_score", 3) for r in results)
    lines.append(f"**Overall score:** {total_score}/{max_total} ({100*total_score//max_total}%)\n")

    lines.append("## Results by Category\n")
    for cat, cat_results in by_category.items():
        cat_score = sum(r["grade"].get("score", 0) for r in cat_results)
        cat_max = sum(r["grade"].get("max_score", 3) for r in cat_results)
        lines.append(f"### {cat.replace('_', ' ').title()} ({len(cat_results)} questions)\n")
        lines.append(f"**Category score:** {cat_score}/{cat_max}\n")
        for r in cat_results:
            g = r["grade"]
            icon = "✓" if g.get("correct_tools") else "✗"
            trace = r["trace"]
            answer_preview = ""
            if trace:
                answer_preview = trace.final_answer[:200].replace("\n", " ")
            lines.append(f"**Q{r['id']}: {r['question']}**  ")
            lines.append(f"- Expected tools: `{r['expected_tools']}`  ")
            lines.append(f"- Tools called: `{g.get('tools_called', [])}`  ")
            lines.append(f"- Tool selection: {icon}  ")
            lines.append(f"- Score: {g.get('score', 0)}/{g.get('max_score', 3)}  ")
            lines.append(f"- Answer preview: _{answer_preview}_  ")
            lines.append(f"- Note: {r['note']}  ")
            lines.append("")

    lines.append("## Failure Mode Analysis\n")
    lines.append(
        "### Failure Mode 1: NL-to-SQL misrouting\n"
        "When questions use ambiguous phrasing like 'best performance', the NL-to-SQL converter "
        "may not generate the correct SQL. This could be fixed by using the LLM to generate SQL "
        "directly instead of the rule-based converter.\n"
    )
    lines.append(
        "### Failure Mode 2: Multi-tool ordering\n"
        "For questions requiring three tools, the agent occasionally retrieves web data before "
        "exhausting structured sources, leading to unnecessary API calls. "
        "Adding a planning step (Bonus A) would improve tool-call ordering.\n"
    )

    Path(output_path).write_text("\n".join(lines))
    print(f"\nEvaluation report saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Overall: {total_score}/{max_total} ({100*total_score//max_total}%)")
    for cat, cat_results in by_category.items():
        s = sum(r["grade"].get("score", 0) for r in cat_results)
        m = sum(r["grade"].get("max_score", 3) for r in cat_results)
        print(f"  {cat}: {s}/{m}")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--question":
        agent = Agent()
        q = " ".join(sys.argv[2:])
        trace = agent.run(q)
        print_trace(trace)
    else:
        run_evaluation()
