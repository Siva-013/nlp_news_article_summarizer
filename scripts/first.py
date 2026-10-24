#!/usr/bin/env python
"""
Display the **first** processed article in
dataset/processed/cnn_dailymail_test_1000_processed.jsonl
"""

import json
from pathlib import Path

def main() -> None:
    jsonl_path = Path(
        "dataset/processed/cnn_dailymail_test_1000_processed.jsonl"
    )

    # Read only the first line (the first article)
    with jsonl_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        print("❌  The file appears to be empty.")
        return

    article = json.loads(first_line)

    # Pretty‑print the whole JSON object
    print(json.dumps(article, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()