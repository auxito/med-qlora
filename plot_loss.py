import os
import json
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def load_log_history(trainer_state_path: str):
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    log_history = data.get("log_history", [])
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    return (train_steps, train_losses), (eval_steps, eval_losses)

def find_trainer_state(output_dir: str):
    root_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.isfile(root_path):
        return root_path
    
    candidates = []
    if not os.path.isdir(output_dir): return None
    for name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, name)
        if os.path.isdir(full_path) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
                candidates.append((step, full_path))
            except ValueError:
                continue
    if not candidates: return None
    candidates.sort(key=lambda x: x[0])
    return os.path.join(candidates[-1][1], "trainer_state.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1_dir", type=str, default="./checkpoints/qwen3-med-qlora-E1")
    parser.add_argument("--e2_dir", type=str, default="./checkpoints/qwen3-med-qlora-E2")
    parser.add_argument("--e3_dir", type=str, default="./checkpoints/qwen3-8b-med-qlora")
    parser.add_argument("--out_path", type=str, default="./outputs/analysis/loss.png")
    
    # === 放大区域参数 ===
    # 注意：Train loss 可能比 Eval loss 低，所以 Y 轴范围可能要设大一点
    parser.add_argument("--zoom_x_min", type=float, default=1000)
    parser.add_argument("--zoom_x_max", type=float, default=2500)
    parser.add_argument("--zoom_y_min", type=float, default=1.38) # 稍微调低一点以容纳 Train loss
    parser.add_argument("--zoom_y_max", type=float, default=1.52) # 稍微调高一点
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    exps = [
        ("4B-E1", args.e1_dir, "tab:blue"),
        ("4B-E2", args.e2_dir, "tab:orange"),
        ("8B-E3", args.e3_dir, "tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # 创建局部放大子图
    axins = ax.inset_axes([0.45, 0.35, 0.5, 0.4]) # [x, y, width, height]

    for label, exp_dir, color in exps:
        ts_path = find_trainer_state(exp_dir)
        if not ts_path:
            print(f"Skipping {exp_dir}")
            continue
            
        train_data, eval_data = load_log_history(ts_path)
        train_steps, train_losses = train_data
        eval_steps, eval_losses = eval_data

        # === 1. 绘制 Train Loss (虚线) ===
        if train_steps:
            # 主图：虚线，透明度 0.5 (看得见，但不抢眼)
            ax.plot(train_steps, train_losses, linestyle='--', color=color, 
                    alpha=0.5, linewidth=1, label=f"{label} (Train)")
            
            # 放大图：也要画 Train！
            axins.plot(train_steps, train_losses, linestyle='--', color=color, 
                       alpha=0.6, linewidth=1.5)

        # === 2. 绘制 Eval Loss (实线) ===
        if eval_steps:
            # 主图：实线
            ax.plot(eval_steps, eval_losses, linestyle='-', color=color, 
                    alpha=1.0, linewidth=2, label=f"{label} (Eval)")
            
            # 放大图：也要画 Eval！
            axins.plot(eval_steps, eval_losses, linestyle='-', color=color, 
                       alpha=1.0, linewidth=2)

    # === 主图设置 ===
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Evaluation Loss Curves", fontsize=14)
    
    # 图例：因为有6条线，分成2列显示比较整齐
    ax.legend(loc='upper right', fontsize=9, frameon=True, ncol=1) 
    ax.grid(True, alpha=0.3)

    # === 放大图设置 ===
    axins.set_xlim(args.zoom_x_min, args.zoom_x_max)
    axins.set_ylim(args.zoom_y_min, args.zoom_y_max)
    axins.grid(True, alpha=0.2, linestyle='--')
    axins.tick_params(axis='both', labelsize=8)

    # 连接线
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    plt.tight_layout()
    plt.savefig(args.out_path)
    print(f"[OK] Saved to: {args.out_path}")

if __name__ == "__main__":
    main()