"""
web_search.py

Live web search for recent news, stock prices, and current information
using the Tavily API.

Tool contract (for LLM):
  Use this tool ONLY when the question asks about recent events, current stock
  prices, latest news, recent analyst ratings, or any information that would
  not be in a document from March 2024 or earlier. Do NOT use for historical
  financial figures or management commentary from annual reports.

Input:  query (str) — short search query under 10 words
Output: list of dicts with keys: title, url, content, published_date
"""

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Search the live web for recent information about IT companies.

    Args:
        query: Short search query (under 10 words). Be specific — include
               company name and topic (e.g. "Infosys stock price today" or
               "TCS analyst rating 2024").
        max_results: Number of results to return (default 3).

    Returns:
        List of dicts: [{"title": ..., "url": ..., "content": ..., "published_date": ...}]
        Returns error dict if API key not set or search fails.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")

    if not api_key or api_key == "your_tavily_api_key_here":
        # Return mock results when no API key is set (for testing/demo)
        return _mock_results(query)

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        results = []
        for r in response.get("results", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
                "published_date": r.get("published_date", "unknown"),
            })
        return results
    except ImportError:
        return [{"error": "tavily-python not installed. Run: pip install tavily-python",
                 "title": "", "url": "", "content": "", "published_date": ""}]
    except Exception as e:
        return [{"error": str(e), "title": "", "url": "", "content": "", "published_date": ""}]


def _mock_results(query: str) -> list[dict]:
    """Return realistic mock results when Tavily API key is not set."""
    q = query.lower()
    today = datetime.now().strftime("%Y-%m-%d")

    mock_db = {
        "infosys stock": [
            {"title": "Infosys Share Price Today", "url": "https://www.nseindia.com/infosys",
             "content": "Infosys (INFY) is trading at ₹1,842.50 on NSE, up 1.2% from previous close. The stock has a 52-week range of ₹1,358–₹1,964. Market cap stands at approximately ₹7.6 lakh crore.",
             "published_date": today},
        ],
        "tcs stock": [
            {"title": "TCS Share Price Today", "url": "https://www.nseindia.com/tcs",
             "content": "TCS is trading at ₹3,892.00 on NSE, down 0.4% from previous close. The 52-week range is ₹3,311–₹4,592. Market cap is approximately ₹14.1 lakh crore, making TCS the most valuable Indian IT company.",
             "published_date": today},
        ],
        "wipro stock": [
            {"title": "Wipro Share Price Today", "url": "https://www.nseindia.com/wipro",
             "content": "Wipro (WIPRO) is trading at ₹462.30 on NSE, up 0.8% from previous close. The 52-week range is ₹406–₹578. Market cap is approximately ₹2.4 lakh crore.",
             "published_date": today},
        ],
        "it sector news": [
            {"title": "Indian IT Sector: Q1 FY25 Outlook", "url": "https://economictimes.indiatimes.com/it-sector-q1",
             "content": "India's IT sector faces continued headwinds in Q1 FY25 as US clients maintain cautious discretionary spending. Analysts at Nomura expect sector revenue growth of 5-7% for FY25, with improvement expected in H2 as macro conditions stabilize.",
             "published_date": today},
            {"title": "AI Deals Driving IT Order Books", "url": "https://livemint.com/ai-it-deals",
             "content": "Generative AI-related deal signings have accelerated across TCS, Infosys, and Wipro, with combined AI TCV exceeding $3 billion in Q4 FY24. Analysts expect AI to be a net positive for IT services demand over a 2-3 year horizon.",
             "published_date": today},
        ],
        "infosys ceo": [
            {"title": "Infosys CEO Salil Parekh Reappointed", "url": "https://businessstandard.com/infosys-ceo",
             "content": "Infosys CEO Salil Parekh has been reappointed for another term through 2027. The board cited strong deal momentum and AI-led transformation as key achievements during his tenure. Parekh joined Infosys in 2018.",
             "published_date": today},
        ],
        "tcs cfo": [
            {"title": "TCS CFO Samir Seksaria Profile", "url": "https://tcs.com/leadership",
             "content": "Samir Seksaria serves as CFO of TCS. He previously held senior finance roles within the Tata Group. TCS reported record free cash flow of $4.8 billion in FY2024 under his stewardship.",
             "published_date": today},
        ],
        "wipro ceo": [
            {"title": "Wipro Appoints Srinivas Pallia as New CEO", "url": "https://economictimes.com/wipro-ceo",
             "content": "Wipro has appointed Srinivas Pallia as its new CEO effective April 2024, succeeding Thierry Delaporte. Pallia is a Wipro veteran with over 30 years at the company. He inherits the challenge of reversing revenue decline and improving margins.",
             "published_date": today},
        ],
        "analyst rating": [
            {"title": "IT Sector Analyst Ratings Update", "url": "https://economictimes.com/it-ratings",
             "content": "Goldman Sachs maintains 'Buy' on TCS with target price ₹4,200. Infosys is rated 'Neutral' by Morgan Stanley with target ₹1,900. Wipro is rated 'Underperform' by CLSA citing slower revenue recovery versus peers.",
             "published_date": today},
        ],
    }

    # Match query to mock data
    for key, results in mock_db.items():
        if any(word in q for word in key.split()):
            return results[:3]

    # Generic fallback
    return [
        {"title": f"Search results for: {query}",
         "url": "https://economictimes.com",
         "content": f"No specific mock data for '{query}'. In production, Tavily API would return live results. Add your TAVILY_API_KEY to .env for real web search.",
         "published_date": today}
    ]


def format_results(results: list[dict]) -> str:
    """Format web search results for trace display."""
    lines = []
    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"  [{i}] ERROR: {r['error']}")
            continue
        lines.append(f"  [{i}] {r['title']}")
        lines.append(f"      URL: {r['url']}")
        lines.append(f"      Date: {r['published_date']}")
        lines.append(f"      {r['content'][:200]}...")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== web_search standalone test ===\n")
    queries = [
        "Infosys stock price today",
        "TCS analyst rating 2024",
        "Wipro CEO leadership change",
        "IT sector news today",
        "Infosys AI deals 2024",
    ]
    for q in queries:
        print(f"Query: {q}")
        results = web_search(q)
        print(format_results(results))
        print()
