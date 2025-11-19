# analyze/compare_4b_best_vs_8b.py
import json
from pathlib import Path

# 复用和 text_stats 一样的模板 & 统计函数（直接拷一份，避免相对 import 麻烦）
CUSTOMER_TEMPLATES = [
    "如果您还有其他问题，请随时提问",
    "如果您有其他问题，请随时提问",
    "祝您身体健康",
    "希望对您有帮助",
    "希望对您有所帮助",
]

DISCLAIMER_TEMPLATES = [
    "仅供参考",
    "不构成医疗建议",
    "不能代替专业医生",
    "不能替代医生的面诊",
    "需要结合具体情况",
    "建议及时就医",
]


def get_paths():
    root = Path(__file__).resolve().parents[1]  # hw3/
    outputs_dir = root / "outputs"
    analysis_dir = outputs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return root, outputs_dir, analysis_dir


def char_stats(text: str):
    text = text.strip()
    if not text:
        return 0, 0.0, 0.0
    chars = list(text)
    length = len(chars)
    distinct1 = len(set(chars)) / length
    if length < 2:
        distinct2 = 0.0
    else:
        bigrams = [tuple(chars[i : i + 2]) for i in range(length - 1)]
        distinct2 = len(set(bigrams)) / len(bigrams)
    return length, distinct1, distinct2


def contains_any(text: str, phrases):
    return any(p in text for p in phrases)


def load_compare_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def summarize_4b_vs_8b(data):
    n = 0
    stats_4b = []
    stats_8b = []

    verdict_counts = {"4b": 0, "8b": 0, "tie": 0}

    for item in data:
        a4 = item["answer_4b_best"]
        a8 = item["answer_8b_med"]
        v = item["verdict"]

        n += 1
        if v == "base":
            verdict_counts["4b"] += 1
        elif v == "finetune":
            verdict_counts["8b"] += 1
        else:
            verdict_counts["tie"] += 1

        l4, d14, d24 = char_stats(a4)
        c4 = contains_any(a4, CUSTOMER_TEMPLATES)
        d4 = contains_any(a4, DISCLAIMER_TEMPLATES)
        stats_4b.append((l4, d14, d24, c4, d4))

        l8, d18, d28 = char_stats(a8)
        c8 = contains_any(a8, CUSTOMER_TEMPLATES)
        d8 = contains_any(a8, DISCLAIMER_TEMPLATES)
        stats_8b.append((l8, d18, d28, c8, d8))

    def avg(lst, idx):
        if not lst:
            return 0.0
        return sum(x[idx] for x in lst) / len(lst)

    def ratio(lst, idx):
        if not lst:
            return 0.0
        return sum(1 for x in lst if x[idx]) / len(lst)

    res = {
        "n": n,
        "verdict": verdict_counts,
        "4b": {
            "avg_len": avg(stats_4b, 0),
            "avg_d1": avg(stats_4b, 1),
            "avg_d2": avg(stats_4b, 2),
            "customer_ratio": ratio(stats_4b, 3),
            "disclaimer_ratio": ratio(stats_4b, 4),
        },
        "8b": {
            "avg_len": avg(stats_8b, 0),
            "avg_d1": avg(stats_8b, 1),
            "avg_d2": avg(stats_8b, 2),
            "customer_ratio": ratio(stats_8b, 3),
            "disclaimer_ratio": ratio(stats_8b, 4),
        },
    }
    return res


def print_summary(res):
    print(f"Total questions: {res['n']}")
    print("Win counts (4B-best / 8B-med / tie):", res["verdict"])
    print()
    print("| Model | AvgLen | Dist-1 | Dist-2 | Customer% | Disclaimer% |")
    print("|-------|--------|--------|--------|-----------|-------------|")
    for m in ["4b", "8b"]:
        s = res[m]
        print(
            f"| {m:5s} | "
            f"{s['avg_len']:.1f} | "
            f"{s['avg_d1']:.3f} | "
            f"{s['avg_d2']:.3f} | "
            f"{s['customer_ratio']*100:9.1f}% | "
            f"{s['disclaimer_ratio']*100:11.1f}% |"
        )


def main():
    _, outputs_dir, _ = get_paths()
    path = outputs_dir / "eval_4b_best_vs_8b_med.json"
    if not path.exists():
        print(f"[ERR] compare file not found: {path}")
        return
    data = load_compare_file(path)
    res = summarize_4b_vs_8b(data)
    print_summary(res)


if __name__ == "__main__":
    main()
