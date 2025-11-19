import json
import os

def merge_files():
    # 定义文件路径
    # 输入文件在 outputs 文件夹中
    file_path_4b = os.path.join('outputs', 'eval_E1_med.json')
    file_path_8b = os.path.join('outputs', 'eval_8B_med.json')
    
    # 输出文件在根目录
    output_file = 'merged_med_comparison.json'

    # 检查文件是否存在
    if not os.path.exists(file_path_4b) or not os.path.exists(file_path_8b):
        print(f"错误: 请确保 '{file_path_4b}' 和 '{file_path_8b}' 文件存在。")
        return

    try:
        # 读取文件
        print(f"正在读取 {file_path_4b}...")
        with open(file_path_4b, 'r', encoding='utf-8') as f:
            data_4b = json.load(f)

        print(f"正在读取 {file_path_8b}...")
        with open(file_path_8b, 'r', encoding='utf-8') as f:
            data_8b = json.load(f)

        # 将 8B 数据转换为以 idx 为 key 的字典，方便快速查找
        # 防止两个文件顺序不一致导致的数据错乱
        data_8b_map = {item['idx']: item for item in data_8b}

        merged_list = []

        # 遍历 4B 数据 (作为主基准)
        for item_4b in data_4b:
            idx = item_4b.get('idx')
            
            # 获取对应的 8B 数据
            item_8b = data_8b_map.get(idx)

            if item_8b:
                # 构建新的数据对象
                new_entry = {
                    "idx": idx,
                    "question": item_4b.get('question', ''),
                    "reference_answer": item_4b.get('reference_answer', ''),
                    # 提取 eval_E1_med.json 的 finetune_answer
                    "answer_4b_best": item_4b.get('finetune_answer', ''),
                    # 提取 eval_8B_med.json 的 finetune_answer
                    "answer_8b_med": item_8b.get('finetune_answer', ''),
                    # 预留 verdict 字段，默认为空字符串
                    "verdict": "" 
                }
                merged_list.append(new_entry)
            else:
                print(f"警告: 在 8B 文件中未找到 idx 为 {idx} 的数据")

        # 写入结果文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文正常显示，indent=2 用于美化格式
            json.dump(merged_list, f, ensure_ascii=False, indent=2)

        print(f"成功! 已合并 {len(merged_list)} 条数据。")
        print(f"输出文件已保存为: {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    merge_files()