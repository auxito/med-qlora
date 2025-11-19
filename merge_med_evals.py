import json
import os

def merge_files():
    file_path_4b = os.path.join('outputs', 'eval_E1_med.json')
    file_path_8b = os.path.join('outputs', 'eval_8B_med.json')
    
    output_file = 'merged_med_comparison.json'

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

        data_8b_map = {item['idx']: item for item in data_8b}

        merged_list = []

        for item_4b in data_4b:
            idx = item_4b.get('idx')
            
            item_8b = data_8b_map.get(idx)

            if item_8b:
                new_entry = {
                    "idx": idx,
                    "question": item_4b.get('question', ''),
                    "reference_answer": item_4b.get('reference_answer', ''),
                    "answer_4b_best": item_4b.get('finetune_answer', ''),
                    "answer_8b_med": item_8b.get('finetune_answer', ''),
                    "verdict": "" 
                }
                merged_list.append(new_entry)
            else:
                print(f"警告: 在 8B 文件中未找到 idx 为 {idx} 的数据")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=2)

        print(f"输出文件已保存为: {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    merge_files()