"""Sample source passages for writing the evaluation questions (M4).

Picks, per topic, passages that state concrete findings: sections abstract / results /
discussion / conclusion, 120-320 words, containing a number, at most one per article.
The sample is seeded, so it can be re-drawn exactly. Output (git-ignored) is a Markdown
file of passages to read while writing eval/qa/questions.json.

Usage: python scripts/sample_qa_passages.py --per-topic 14 --out data/eval/candidates.md
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SECTIONS = {"abstract", "results", "discussion", "conclusion"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--per-topic", type=int, default=14)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("data/eval/candidates.md"))
    args = parser.parse_args()

    by_topic: dict[str, list[dict]] = defaultdict(list)
    with args.chunks.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            n = len(c["text"].split())
            if c["section"] in SECTIONS and 120 <= n <= 320 and re.search(r"\d", c["text"]):
                by_topic[c["topic"]].append(c)

    rng = random.Random(args.seed)
    lines = []
    for topic in sorted(by_topic):
        pool = by_topic[topic]
        rng.shuffle(pool)
        seen_docs: set[str] = set()
        picked = 0
        for c in pool:
            if c["doc_id"] in seen_docs:
                continue
            seen_docs.add(c["doc_id"])
            lines += [
                f"## {c['chunk_id']} | {topic} | {c['section']} | {c['year']}",
                f"**{c['title']}**",
                "",
                c["text"],
                "",
            ]
            picked += 1
            if picked == args.per_topic:
                break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {sum(1 for x in lines if x.startswith('## '))} passages to {args.out}")


if __name__ == "__main__":
    main()
