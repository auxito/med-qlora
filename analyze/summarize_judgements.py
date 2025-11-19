import json
from collections import Counter
from pathlib import Path

EXP_NAME = {
    "4b_E1": "4B-E1 (lr=2e-4)",
    "4b_E2": "4B-E2 (lr=1e-4)",
    "8b_E3": "8B-E3 (lr=1e-4)",
    "4b_best_vs_8b": "4B-best vs 8B-med",
}


def get_paths():
    """统一管理路径，保证无论从哪里运行都OK"""
    root = Path(__file__).resolve().parents[1]  
    analyze_dir = root / "analyze"
    return root, analyze_dir


def load_judgements():
    _, analyze_dir = get_paths()
    path = analyze_dir / "judge_results_med.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def summarize_counts(judgements):
    summary = {}
    for exp_key, items in judgements.items():
        counts = Counter()
        for item in items:
            v = item["verdict"]
            if v not in {"base", "finetune", "tie"}:
                raise ValueError(f"Unexpected verdict: {v}")
            counts[v] += 1
        summary[exp_key] = counts
    return summary


def print_markdown_table(summary):
    print("| Experiment | Base Win | Finetune Win | Tie | Total |")
    print("|-----------|----------|--------------|-----|-------|")
    for exp_key, counts in summary.items():
        total = counts["base"] + counts["finetune"] + counts["tie"]
        name = EXP_NAME.get(exp_key, exp_key)
        print(
            f"| {name} | "
            f"{counts['base']} | "
            f"{counts['finetune']} | "
            f"{counts['tie']} | "
            f"{total} |"
        )


def main():
    judgements = load_judgements()
    summary = summarize_counts(judgements)
    print_markdown_table(summary)


if __name__ == "__main__":
    main()
