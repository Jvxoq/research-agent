"""Draw the M7 chart: prompt tokens per round, report arm against full history.

Both arms ran the same BrowseComp question, so each line is one run of it.
The report arm (M6) sends a short running report each round. The full history
arm (M7) sends every past observation, so its prompt grows with the run.

Run from backend/:

    uv run --with matplotlib python -m evals.research_agent.plot_m7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BACKEND = Path(__file__).resolve().parents[2]
REPORT_RESULTS = BACKEND / "results/m6/browsecomp.jsonl"
FULL_HISTORY_RESULTS = BACKEND / "results/m7/full_history.jsonl"
DEFAULT_OUT = BACKEND.parent / "docs/m7_prompt_tokens.png"
DEFAULT_QUESTION = "browsecomp-776"


def load_run(path: Path, question_id: str) -> dict[str, Any]:
    """Return the result line for one question from a results file."""
    with path.open() as f:
        for line in f:
            run = json.loads(line)
            if run["question_id"] == question_id:
                return run
    raise SystemExit(f"{question_id} is not in {path}")


def main() -> None:
    """Read both runs and save the chart as a PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    arms = [
        ("Full history (M7)", load_run(FULL_HISTORY_RESULTS, args.question), "#d1495b"),
        ("Running report (M6)", load_run(REPORT_RESULTS, args.question), "#2e86ab"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for label, run, color in arms:
        rounds = [r["round"] + 1 for r in run["per_round"]]
        prompt_tokens = [r["prompt_tokens"] for r in run["per_round"]]
        total = sum(prompt_tokens)
        ax.plot(
            rounds,
            prompt_tokens,
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{label}: {total:,} prompt tokens in total",
        )
        ax.annotate(
            f"{prompt_tokens[-1]:,}",
            (rounds[-1], prompt_tokens[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            color=color,
            fontsize=9,
        )

    report_total = sum(r["prompt_tokens"] for r in arms[1][1]["per_round"])
    history_total = sum(r["prompt_tokens"] for r in arms[0][1]["per_round"])
    ax.set_title(
        f"Prompt size per round on {args.question}: the running report sends "
        f"{history_total / report_total:.1f}x fewer prompt tokens",
        fontsize=11,
        loc="left",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Prompt tokens sent to the model")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xticks(range(1, max(len(run["per_round"]) for _, run, _ in arms) + 1))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
