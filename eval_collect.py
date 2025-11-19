# eval_collect.py  —— 医学问答评测版
import os
import json
import argparse
import torch
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

BASE_MODEL = os.path.expanduser("~/models/Qwen3-8B")


def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data: List[[question, reference_answer], ...]
    questions = []
    for idx, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        q = item[0]
        ref = item[1] if len(item) > 1 else ""
        questions.append({"idx": idx, "question": q, "reference_answer": ref})
    print(f">>> 从 {path} 读取到 {len(questions)} 个问题")
    return questions


def build_prompt(tokenizer, question: str) -> str:
    """用 chat 模板构造“医学助手”风格提示。"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名专业、严谨的中文医学知识助手，擅长向非专业大众解释医学概念、疾病、"
                "诊疗流程和公共卫生知识。回答时要尽量："
                "1）内容准确、基于常识医学知识；"
                "2）语言通俗易懂；"
                "3）避免给出具体用药剂量或个体化诊疗方案，必要时提醒患者咨询专业医生。"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def generate_answers(
    tokenizer,
    model,
    questions: List[Dict],
    max_new_tokens: int = 256,
    temperature: float = 0.4,
    top_p: float = 0.9,
) -> List[str]:
    """生成一批答案，稍微保守一点的采样参数，避免太啰嗦/复读。"""
    answers = []
    device = model.device
    model.eval()

    for i, item in enumerate(questions):
        q = item["question"]
        prompt = build_prompt(tokenizer, q)
        inputs = tokenizer([prompt], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,  # 抑制“复读机”
            )

        # 只取新生成的部分
        output_ids = generated_ids[0][len(inputs.input_ids[0]):]
        ans = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        answers.append(ans)

        print(f"[{i+1}/{len(questions)}] 问题: {q[:20]}... → 答案长度: {len(ans)} 字")

    return answers


def load_4bit_base_model():
    print(">>> 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False,
        local_files_only=True,   # ✅ 只用本地，不联网
    )
    tokenizer.pad_token = tokenizer.eos_token

    print(">>> 以 4bit 方式加载基座模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,  
    )

    return tokenizer, model


def load_4bit_lora_model(adapter_dir: str):
    """基于相同 base 模型 + LoRA 适配器加载 finetune 模型。"""
    print(">>> 从 LoRA 目录加载 tokenizer（本地）...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(">>> 以 4bit 方式加载基座模型（本地缓存）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    )

    print(f">>> 从 {adapter_dir} 加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        local_files_only=True,
    )
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions_path",
        type=str,
        default="zh_med.json",  # ✅ 改成医学问题文件
        help="老师提供的 20 个医学问题的 JSON 文件路径",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="checkpoints/qwen3-med-qlora-3090",
        help="你训练好的 LoRA 适配器目录 (OUTPUT_DIR)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="eval_E1_med.json",  # ✅ 医学版输出
        help="保存 baseline / finetune 回答的 JSON 文件路径",
    )
    args = parser.parse_args()

    # 1. 读问题
    questions = load_questions(args.questions_path)

    # 2. baseline: 原始 Qwen3-4B
    print("\n=== 生成 baseline（原始 Qwen3-4B）回答 ===")
    tokenizer_base, model_base = load_4bit_base_model()
    base_answers = generate_answers(tokenizer_base, model_base, questions)

    # 释放显存
    del model_base
    torch.cuda.empty_cache()

    # 3. finetune: 你训练好的 LoRA 模型
    print("\n=== 生成 finetune（Qwen3-4B-med-QLoRA）回答 ===")
    tokenizer_ft, model_ft = load_4bit_lora_model(args.adapter_dir)
    ft_answers = generate_answers(tokenizer_ft, model_ft, questions)

    # 4. 汇总 & 保存
    results = []
    for item, base_ans, ft_ans in zip(questions, base_answers, ft_answers):
        results.append(
            {
                "idx": item["idx"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "base_model": BASE_MODEL,
                "base_answer": base_ans,
                "finetune_adapter_dir": args.adapter_dir,
                "finetune_answer": ft_ans,
            }
        )

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n>>> 已保存 evaluation 数据到: {args.output_path}")
    print("你可以把其中的 question / base_answer / finetune_answer 拿去让 ChatGPT 做主观评价。")


if __name__ == "__main__":
    main()
