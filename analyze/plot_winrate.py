import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

EXP_NAME = {
    "4b_E1": "4B-E1",
    "4b_E2": "4B-E2",
    "8b_E3": "8B-E3",
    "4b_best_vs_8b": "4B-best vs 8B",
}


def get_paths():
    root = Path(__file__).resolve().parents[1]  
    analyze_dir = root / "analyze"
    outputs_dir = root / "outputs"
    analysis_dir = outputs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return root, analyze_dir, outputs_dir, analysis_dir


def load_judgements():
    _, analyze_dir, _, _ = get_paths()
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


def plot_winrate(summary, save_path):
    exp_keys = ["4b_E1", "4b_E2", "8b_E3", "4b_best_vs_8b"]
    exp_keys = [k for k in exp_keys if k in summary]

    labels = [EXP_NAME.get(k, k) for k in exp_keys]

    base_wins = [summary[k]["base"] for k in exp_keys]
    ft_wins = [summary[k]["finetune"] for k in exp_keys]
    ties = [summary[k]["tie"] for k in exp_keys]

    x = range(len(exp_keys))
    width = 0.25

    plt.figure()
    plt.bar([i - width for i in x], base_wins, width, label="Base Win")
    plt.bar(x, ft_wins, width, label="Finetune Win")
    plt.bar([i + width for i in x], ties, width, label="Tie")

    plt.xticks(list(x), labels)
    plt.ylabel("Count (out of 20)")
    plt.title("ChatGPT Judgement on 20 Medical Questions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    _, _, _, analysis_dir = get_paths()
    judgements = load_judgements()
    summary = summarize_counts(judgements)
    out_path = analysis_dir / "winrate_med.png"
    plot_winrate(summary, out_path)
    print(f"[OK] winrate plot saved to: {out_path}")


if __name__ == "__main__":
    main()
