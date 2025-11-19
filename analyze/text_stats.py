# analyze/text_stats.py
import json
from pathlib import Path

import matplotlib.pyplot as plt

# 实验到 eval 文件的映射（注意路径在 outputs/）
EVAL_FILES = {
    "4b_E1": "eval_E1_med.json",
    "4b_E2": "eval_E2_med.json",
    "8b_E3": "eval_8B_med.json",
}

EXP_LABEL = {
    "4b_E1": "4B-E1",
    "4b_E2": "4B-E2",
    "8b_E3": "8B-E3",
}

CUSTOMER_TEMPLATES = [
    "如果您还有其他问题",
    "如果您还有任何疑问",
    "请随时提问",
    "请随时告诉我",
    "希望对您有帮助",
    "希望对您有所帮助",
    "希望我的回答",
    "希望这个回答",
    "祝您身体健康",
    "祝您健康",
    "祝您生活愉快",
    "谢谢！",
    "😊",               # 4B-E2 模型非常喜欢用 emoji
    "我会尽力",
    "如果有其他问题",
    "如果有任何疑问",
]

DISCLAIMER_TEMPLATES = [
    "仅供参考",
    "不能替代医生",
    "不能代替医生",
    "不构成医疗建议",
    "建议咨询",
    "请咨询医生",
    "具体情况请",
    "以当地卫生部门",  # 针对 8B 模型中出现的“以当地...为准”
    "注：",            # 很多免责声明以“注：”开头
    "注意：",
    "遵医嘱",
]


def get_paths():
    root = Path(__file__).resolve().parents[1]  # hw3/
    outputs_dir = root / "outputs"
    analysis_dir = outputs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return root, outputs_dir, analysis_dir


def load_eval_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容 {"results": [...]} 的格式
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    return data


def char_stats(text: str):
    """返回 (长度, distinct-1, distinct-2)"""
    text = text.strip()
    if not text:
        return 0, 0.0, 0.0

    chars = list(text)
    length = len(chars)

    # distinct-1
    distinct1 = len(set(chars)) / length

    # distinct-2
    if length < 2:
        distinct2 = 0.0
    else:
        bigrams = [tuple(chars[i : i + 2]) for i in range(length - 1)]
        distinct2 = len(set(bigrams)) / len(bigrams)

    return length, distinct1, distinct2


def contains_any(text: str, phrases):
    return any(p in text for p in phrases)


def collect_stats_for_model(exp_key, eval_path: Path):
    data = load_eval_file(eval_path)

    base_lengths, base_d1, base_d2 = [], [], []
    ft_lengths, ft_d1, ft_d2 = [], [], []
    base_customer = base_disclaimer = 0
    ft_customer = ft_disclaimer = 0
    n = 0

    for item in data:
        # 字段名做兼容处理
        base_ans = (
            item.get("base_answer")
            or item.get("base_answer_4b")
            or item.get("base_answer_8b")
        )
        ft_ans = (
            item.get("finetune_answer")
            or item.get("finetune_answer_4b")
            or item.get("finetune_answer_8b")
        )
        if base_ans is None or ft_ans is None:
            continue

        n += 1

        bl, bd1, bd2 = char_stats(base_ans)
        base_lengths.append(bl)
        base_d1.append(bd1)
        base_d2.append(bd2)
        if contains_any(base_ans, CUSTOMER_TEMPLATES):
            base_customer += 1
        if contains_any(base_ans, DISCLAIMER_TEMPLATES):
            base_disclaimer += 1

        fl, fd1, fd2 = char_stats(ft_ans)
        ft_lengths.append(fl)
        ft_d1.append(fd1)
        ft_d2.append(fd2)
        if contains_any(ft_ans, CUSTOMER_TEMPLATES):
            ft_customer += 1
        if contains_any(ft_ans, DISCLAIMER_TEMPLATES):
            ft_disclaimer += 1

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    stats = {
        "num_samples": n,
        "base": {
            "avg_len": avg(base_lengths),
            "avg_d1": avg(base_d1),
            "avg_d2": avg(base_d2),
            "customer_ratio": base_customer / n if n else 0.0,
            "disclaimer_ratio": base_disclaimer / n if n else 0.0,
        },
        "finetune": {
            "avg_len": avg(ft_lengths),
            "avg_d1": avg(ft_d1),
            "avg_d2": avg(ft_d2),
            "customer_ratio": ft_customer / n if n else 0.0,
            "disclaimer_ratio": ft_disclaimer / n if n else 0.0,
        },
    }
    return stats


def print_stats_table(all_stats):
    print(
        "| Exp | Model     | AvgLen | Dist-1 | Dist-2 | Customer% | Disclaimer% |"
    )
    print(
        "|-----|-----------|--------|--------|--------|-----------|-------------|"
    )
    for exp_key, stats in all_stats.items():
        label = EXP_LABEL.get(exp_key, exp_key)
        for which in ["base", "finetune"]:
            s = stats[which]
            print(
                f"| {label} | {which:9s} | "
                f"{s['avg_len']:.1f} | "
                f"{s['avg_d1']:.3f} | "
                f"{s['avg_d2']:.3f} | "
                f"{s['customer_ratio']*100:9.1f}% | "
                f"{s['disclaimer_ratio']*100:11.1f}% |"
            )


def plot_avg_lengths(all_stats, save_path: Path):
    exps = list(all_stats.keys())
    labels = [EXP_LABEL.get(k, k) for k in exps]

    base_lens = [all_stats[k]["base"]["avg_len"] for k in exps]
    ft_lens = [all_stats[k]["finetune"]["avg_len"] for k in exps]

    x = range(len(exps))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], base_lens, width, label="Base")
    plt.bar([i + width / 2 for i in x], ft_lens, width, label="Finetune")

    plt.xticks(list(x), labels)
    plt.ylabel("Average length (chars)")
    plt.title("Average Answer Length per Experiment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    _, outputs_dir, analysis_dir = get_paths()

    all_stats = {}
    for exp_key, fname in EVAL_FILES.items():
        path = outputs_dir / fname
        if not path.exists():
            print(f"[WARN] Eval file not found for {exp_key}: {path}")
            continue
        stats = collect_stats_for_model(exp_key, path)
        all_stats[exp_key] = stats

    print_stats_table(all_stats)
    out_img = analysis_dir / "avg_lengths.png"
    plot_avg_lengths(all_stats, out_img)
    print(f"[OK] avg length plot saved to: {out_img}")


if __name__ == "__main__":
    main()
