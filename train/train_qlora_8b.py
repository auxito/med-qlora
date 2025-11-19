# train_qlora_8b.py
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

# ===== 1. 基座模型 & 数据路径 =====

# 本地 Qwen3-8B 模型路径（使用 ~ 展开）
MODEL_PATH = os.path.expanduser("~/models/Qwen3-8B")

# 已经用 prepare_data.py 处理好的数据集
DATASET_DISK_PATH = "data/huatuo_qwen3"

# 8B 实验的新输出目录
OUTPUT_DIR = "checkpoints/qwen3-8b-med-qlora"


# ===== 2. QLoRA 量化配置 =====
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ===== 3. LoRA 配置 =====
def get_lora_config():
    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


# ===== 4. tokenize 函数 =====
def tokenize_function(examples, tokenizer, max_length: int):
    # 假设数据集有 "text" 字段
    return tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 小优化：开启 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ----- 1. 加载 tokenizer & 数据 -----
    print(">>> [8B] 加载 tokenizer 和数据集...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512

    datasets = load_from_disk(DATASET_DISK_PATH)

    MAX_TRAIN_SAMPLES = 40_000
    MAX_EVAL_SAMPLES = 2_000

    full_train = datasets["train"]
    full_eval = datasets["validation"]

    if len(full_train) > MAX_TRAIN_SAMPLES:
        train_dataset = full_train.select(range(MAX_TRAIN_SAMPLES))
    else:
        train_dataset = full_train

    if len(full_eval) > MAX_EVAL_SAMPLES:
        eval_dataset = full_eval.select(range(MAX_EVAL_SAMPLES))
    else:
        eval_dataset = full_eval

    print(f"[8B] 训练集样本数: {len(train_dataset)}")
    print(f"[8B] 验证集样本数: {len(eval_dataset)}")

    max_length = tokenizer.model_max_length
    print(f">>> [8B] 将文本 tokenize，max_length={max_length} ...")

    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # ----- 2. 加载 8B 4bit 模型 -----
    print(">>> [8B] 以 4bit 量化方式加载基座模型（Qwen3-8B, QLoRA）...")
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.use_cache = False  # 配合 gradient_checkpointing

    # ----- 3. 注入 LoRA -----
    print(">>> [8B] 注入 LoRA 适配器...")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 关键：让输入参与梯度计算，避免 "does not require grad" 报错
    model.enable_input_require_grads()

    # ----- 4. Data collator -----
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ----- 5. 训练参数（适配 8B + 3090） -----
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,   # 单卡 2，比 4B 更保守
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,   # 有效 batch = 2*8=16
        num_train_epochs=1.0,            
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to=[],
    )

    # ----- 6. Trainer + 断点续训 -----
    print(">>> [8B] 创建 Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint is not None:
            print(f">>> [8B] 检测到已有 checkpoint, 将从 {last_checkpoint} 恢复训练...")
        else:
            print(">>> [8B] 未检测到现有 checkpoint, 将从头开始训练...")
    else:
        print(">>> [8B] 输出目录不存在, 将从头开始训练...")

    print(">>> [8B] 开始训练 QLoRA (Qwen3-8B, 1 epoch)...")
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # ----- 7. 保存适配器和 tokenizer -----
    print(">>> [8B] 保存 LoRA 适配器和 tokenizer...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("8B 训练完成！LoRA 权重已保存在：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
