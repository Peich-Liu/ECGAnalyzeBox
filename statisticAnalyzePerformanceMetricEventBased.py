import pandas as pd

# 文件路径
resultFile =  r"c:\Document\sc2024\filtered_ecg_with_qualitynewwithoutFilter.csv"
annotations_file_path = r'c:\Document\sc2024\data\testData\100001_ANN.csv'

# 读取 ANN 文件和 filtered 文件
ann_df = pd.read_csv(annotations_file_path)
filtered_df = pd.read_csv(resultFile)

# 从 ANN 文件中提取“好”质量的片段
start_col = ann_df.columns[9]  # 第10列：起始样本
end_col = ann_df.columns[10]   # 第11列：结束样本
quality_col = ann_df.columns[11]  # 第12列：质量等级

true_segments = ann_df[ann_df[quality_col] == 1][[start_col, end_col]].values.tolist()
print("true_segments0",true_segments)
# 从 filtered 文件中提取预测的“好”质量片段
filtered_df = filtered_df.sort_values('sample_index').reset_index(drop=True)

pred_segments = []
current_quality = None
start_sample = None

for index, row in filtered_df.iterrows():
    sample_index = row['sample_index']
    quality = row['quality']

    if quality == current_quality:
        # 继续当前片段
        continue
    else:
        # 片段发生变化
        if current_quality == 'good':
            # 结束当前“好”质量片段
            end_sample = sample_index - 1
            pred_segments.append([start_sample, end_sample])
        if quality == 'good':
            # 开始新的“好”质量片段
            start_sample = sample_index
        current_quality = quality

# 处理最后一个片段
if current_quality == 'good':
    end_sample = filtered_df.iloc[-1]['sample_index']
    pred_segments.append([start_sample, end_sample])

# 定义计算重叠的函数
def calculate_overlap(segment1, segment2):
    start = max(segment1[0], segment2[0])
    end = min(segment1[1], segment2[1])
    overlap = max(0, end - start + 1)
    return overlap

def is_significant_overlap(pred_segment, true_segment, threshold=0.5):
    overlap = calculate_overlap(pred_segment, true_segment)
    true_length = true_segment[1] - true_segment[0] + 1
    if true_length == 0:
        return False
    overlap_ratio = overlap / true_length
    return overlap_ratio >= threshold

# 计算 TP 和 FP
TP = 0
FP = 0

for pred_segment in pred_segments:
    match_found = False
    for true_segment in true_segments:
        if is_significant_overlap(pred_segment, true_segment, threshold=0.5):
            match_found = True
            break
    if match_found:
        TP += 1
    else:
        FP += 1

# 计算 FN
FN = 0
for true_segment in true_segments:
    match_found = False
    for pred_segment in pred_segments:
        if is_significant_overlap(pred_segment, true_segment, threshold=0.5):
            match_found = True
            break
    if not match_found:
        FN += 1

# 计算性能指标
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 输出结果
print(f"真阳性 (TP): {TP}")
print(f"假阳性 (FP): {FP}")
print(f"假阴性 (FN): {FN}")
print(f"精确率 (Precision): {precision:.2f}")
print(f"召回率 (Recall): {recall:.2f}")
print(f"F1 分数 (F1 Score): {f1_score:.2f}")
