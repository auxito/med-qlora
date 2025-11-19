# prepare_data.py
import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = "FreedomIntelligence/Huatuo26M-Lite"
OUTPUT_DIR = "data"


def load_and_split():
    print(">>> 加载原始 Huatuo26M-Lite 数据集...")
    ds = load_dataset(DATASET_NAME)["train"]

    # 过滤 score < 4 的样本
    ds = ds.filter(lambda e: e["score"] >= 4)
    ds = ds.shuffle(seed=42)

    print("过滤后样本数:", len(ds))

    # 默认划分：40k train, 2k val, 1k test（如果数据不够就全用）
    train_size = min(40000, max(0, len(ds) - 3000))
    val_size = min(2000, max(0, len(ds) - train_size))
    test_size = min(1000, max(0, len(ds) - train_size - val_size))

    train_ds = ds.select(range(0, train_size))
    val_ds = ds.select(range(train_size, train_size + val_size))
    test_ds = ds.select(range(train_size + val_size,
                              train_size + val_size + test_size))

    print(f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})


def format_with_qwen_template(raw_datasets: DatasetDict):
    print(">>> 加载 tokenizer，用于构造对话模板...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def format_example(example):
        dept = example.get("label", "") or "未知科室"
        disease = example.get("related_diseases", "") or "未标明疾病"
        question = example["question"]
        answer = example["answer"]

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名专业且严谨的中文临床医生助手，需要根据患者的中文描述"
                    "给出科学、谨慎、可执行的建议，但避免给出绝对诊断和具体药物剂量。"
                ),
            },
            {
                "role": "user",
                "content": f"科室：{dept}\n可能相关疾病：{disease}\n患者问题：{question}",
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print(">>> 使用 Qwen3 chat 模板格式化文本...")
    new_datasets = {}
    for split, ds in raw_datasets.items():
        new_datasets[split] = ds.map(
            format_example,
            remove_columns=ds.column_names,
        )
    return DatasetDict(new_datasets)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_datasets = load_and_split()
    formatted = format_with_qwen_template(raw_datasets)

    save_path = os.path.join(OUTPUT_DIR, "huatuo_qwen3")
    print(f">>> 保存到磁盘: {save_path}")
    formatted.save_to_disk(save_path)

    print("完成！数据已保存到:", save_path)


if __name__ == "__main__":
    main()
