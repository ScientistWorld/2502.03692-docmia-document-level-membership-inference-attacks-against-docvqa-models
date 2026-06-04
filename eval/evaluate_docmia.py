"""Evaluate DocMIA outputs without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="checkpoints/docmia_outputs.json")
    parser.add_argument("--scores", default="scoring/scores.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = payload["results"]
    scores = {
        "experiments": {
            "scaled_whitebox_docmia": {
                "results": {
                    name: results[name]
                    for name in ("score_ua", "gradient_ua", "docmia_fl", "docmia_ig")
                    if name in results
                }
            }
        }
    }
    Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
    Path(args.scores).write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
