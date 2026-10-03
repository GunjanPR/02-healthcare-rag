"""Scripted walkthrough of the running API; writes a Markdown transcript.

Usage: python scripts/demo.py --base-url http://127.0.0.1:8010 --markdown docs/demo.md
Needs the full system running (make up, the gateway, make dev with the reranker).
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

STEPS = [
    ("Liveness", "GET", "/health", None),
    ("Readiness: OpenSearch and Qdrant reachable, gateway reported", "GET", "/ready", None),
    (
        "Answerable question: cited answer with PMC links",
        "POST",
        "/ask",
        {
            "question": "Which gyrA mutation was most common among ofloxacin-resistant "
            "multidrug-resistant TB strains in Morocco?"
        },
    ),
    (
        "Paraphrased question that BM25 ranks poorly (relevant passage not in its top 10)",
        "POST",
        "/ask",
        {
            "question": "In children aged 12 and older, is the Asthma Control Test more useful "
            "for ruling out or for confirming exercise-induced bronchoconstriction?"
        },
    ),
    (
        "Out-of-scope question: declined by the rerank-score gate, no LLM call",
        "POST",
        "/ask",
        {"question": "Which bacterium causes Lyme disease?"},
    ),
    (
        "Metadata filters: only results sections of osteoporosis articles from 2020 on",
        "POST",
        "/ask",
        {
            "question": "What share of patients started osteoporosis treatment after a "
            "fracture liaison service was introduced?",
            "filters": {"topics": ["osteoporosis"], "sections": ["results"], "year_min": 2020},
        },
    ),
    ("Summarize one article", "POST", "/summarize", {"pmcid": "PMC10497696"}),
    ("Input validation", "POST", "/ask", {"question": ""}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--markdown", type=Path, default=Path("docs/demo.md"))
    args = parser.parse_args()
    client = httpx.Client(base_url=args.base_url, timeout=180)
    client.post("/ask", json={"question": "warm-up: what is asthma?"})

    out = [
        "# Demo transcript",
        "",
        f"Recorded {datetime.now(UTC):%Y-%m-%d %H:%M UTC} against the full system (hybrid "
        "retrieval, MedCPT rerank, llama3.2 via the LLM gateway) by `scripts/demo.py`. "
        "Responses are real and only trimmed (long snippets shortened). The API uses the "
        "gateway's response cache, so repeated questions can be answered from cache.",
        "",
    ]
    for title, method, path, body in STEPS:
        start = time.perf_counter()
        resp = client.request(method, path, json=body)
        secs = time.perf_counter() - start
        data = resp.json()
        for c in data.get("citations", []) if isinstance(data, dict) else []:
            c["snippet"] = c["snippet"][:160] + "..."
        out += [f"## {title}", "", f"`{method} {path}` -> **{resp.status_code}** in {secs:.2f} s"]
        if body is not None:
            out += ["", "Request:", "```json", json.dumps(body, indent=2), "```"]
        out += [
            "",
            "Response:",
            "```json",
            json.dumps(data, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
        print(f"{resp.status_code} {secs:5.2f}s {title}")
    metrics = client.get("/metrics").text.splitlines()
    keep = [m for m in metrics if m.startswith(("rag_answers_total", "rag_requests_total"))]
    out += ["## Metrics after the demo", "", "`GET /metrics` (excerpt):", "```", *keep, "```", ""]
    args.markdown.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
