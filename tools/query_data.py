"""
query_data.py

Structured query tool over the financials SQLite database.
Accepts natural language questions and converts them to SQL,
or accepts raw SQL directly.

Tool contract (for LLM):
  Use this tool when the question asks for specific numbers, figures, margins,
  revenue, profit, EPS, headcount, or any quantitative comparison across
  companies or years. This tool queries the financials table which has columns:
  company, year, revenue_cr, op_margin_pct, net_profit_cr, eps, headcount.
  Do NOT use for qualitative information or management commentary.

Input:  question (str) — natural language question about financial data
Output: dict with keys: columns, rows, row_count, sql_used
"""

import os
import sqlite3
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "financials.db"
CSV_PATH = DATA_DIR / "financials.csv"


def _build_db() -> None:
    """Load CSV into SQLite if DB doesn't exist."""
    print("[query_data] Building SQLite database from CSV...")
    df = pd.read_csv(CSV_PATH)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("financials", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[query_data] DB built: {len(df)} rows")


def _nl_to_sql(question: str) -> str:
    """
    Rule-based natural language to SQL converter.
    Handles common patterns without needing an LLM call.
    Falls back to a broad SELECT if no pattern matches.
    """
    q = question.lower()

    # Revenue queries
    if re.search(r"revenue.*(infosys|tcs|wipro)", q) or re.search(r"(infosys|tcs|wipro).*revenue", q):
        company = None
        for c in ["infosys", "tcs", "wipro"]:
            if c in q:
                company = c.title()
        if "growth" in q or "trend" in q or "year" in q:
            return f"SELECT year, revenue_cr FROM financials WHERE company='{company}' ORDER BY year"
        year_match = re.search(r"20(2[0-4])", q)
        if year_match and company:
            yr = int("20" + year_match.group(1))
            return f"SELECT revenue_cr FROM financials WHERE company='{company}' AND year={yr}"
        if company:
            return f"SELECT year, revenue_cr FROM financials WHERE company='{company}' ORDER BY year"

    # Margin queries
    if "margin" in q:
        year_match = re.search(r"(20[0-9]{2})", q)
        company = None
        for c in ["infosys", "tcs", "wipro"]:
            if c in q:
                company = c.title()
        if year_match and company:
            yr = year_match.group(1)
            return f"SELECT op_margin_pct FROM financials WHERE company='{company}' AND year={yr}"
        if year_match:
            yr = year_match.group(1)
            return f"SELECT company, op_margin_pct FROM financials WHERE year={yr} ORDER BY op_margin_pct DESC"
        if company:
            return f"SELECT year, op_margin_pct FROM financials WHERE company='{company}' ORDER BY year"
        return "SELECT company, year, op_margin_pct FROM financials ORDER BY year, company"

    # Headcount queries
    if "headcount" in q or "employee" in q or "staff" in q:
        year_match = re.search(r"(20[0-9]{2})", q)
        company = None
        for c in ["infosys", "tcs", "wipro"]:
            if c in q:
                company = c.title()
        if company:
            return f"SELECT year, headcount FROM financials WHERE company='{company}' ORDER BY year"
        if year_match:
            yr = year_match.group(1)
            return f"SELECT company, headcount FROM financials WHERE year={yr} ORDER BY headcount DESC"
        return "SELECT company, year, headcount FROM financials ORDER BY company, year"

    # Net profit
    if "profit" in q or "net profit" in q:
        company = None
        for c in ["infosys", "tcs", "wipro"]:
            if c in q:
                company = c.title()
        if company:
            return f"SELECT year, net_profit_cr FROM financials WHERE company='{company}' ORDER BY year"
        return "SELECT company, year, net_profit_cr FROM financials ORDER BY company, year"

    # EPS
    if "eps" in q or "earnings per share" in q:
        company = None
        for c in ["infosys", "tcs", "wipro"]:
            if c in q:
                company = c.title()
        if company:
            return f"SELECT year, eps FROM financials WHERE company='{company}' ORDER BY year"
        return "SELECT company, year, eps FROM financials ORDER BY company, year"

    # Compare all companies
    if "compare" in q or "all" in q or "three" in q:
        year_match = re.search(r"(20[0-9]{2})", q)
        if year_match:
            yr = year_match.group(1)
            return f"SELECT company, revenue_cr, op_margin_pct, net_profit_cr, headcount FROM financials WHERE year={yr} ORDER BY revenue_cr DESC"
        return "SELECT * FROM financials ORDER BY company, year"

    # Specific year
    year_match = re.search(r"(20[0-9]{2})", q)
    if year_match:
        yr = year_match.group(1)
        return f"SELECT * FROM financials WHERE year={yr} ORDER BY company"

    # Default: return everything
    return "SELECT * FROM financials ORDER BY company, year"


def query_data(question: str, sql: str = None) -> dict:
    """
    Query the financial database with a natural language question or raw SQL.

    Args:
        question: Natural language question about financial figures.
        sql: Optional raw SQL query (overrides NL-to-SQL conversion).

    Returns:
        dict: {
            "columns": list of column names,
            "rows": list of row dicts,
            "row_count": int,
            "sql_used": str
        }
    """
    if not DB_PATH.exists():
        _build_db()

    sql_query = sql if sql else _nl_to_sql(question)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql_query)
        rows = [dict(row) for row in cursor.fetchall()]
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
    except sqlite3.Error as e:
        conn.close()
        return {"error": str(e), "sql_used": sql_query, "rows": [], "columns": [], "row_count": 0}
    finally:
        conn.close()

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "sql_used": sql_query,
    }


def format_result(result: dict) -> str:
    """Format result as a readable table string."""
    if "error" in result:
        return f"Error: {result['error']}"
    if not result["rows"]:
        return "No rows returned."
    header = " | ".join(result["columns"])
    sep = "-" * len(header)
    lines = [header, sep]
    for row in result["rows"]:
        lines.append(" | ".join(str(row.get(c, "")) for c in result["columns"]))
    lines.append(f"\n({result['row_count']} rows) SQL: {result['sql_used']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== query_data standalone test ===\n")
    questions = [
        "What was Infosys operating margin in 2024?",
        "Compare all companies revenue in 2024",
        "Show TCS headcount over all years",
        "What was Wipro net profit in 2023?",
        "Which company had the highest margin in 2024?",
    ]
    for q in questions:
        print(f"Q: {q}")
        result = query_data(q)
        print(format_result(result))
        print()
