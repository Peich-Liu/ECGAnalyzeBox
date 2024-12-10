import pandas as pd
import numpy as np


# filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_105001.csv"
# filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_100001.csv"
filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_103001.csv"


# filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_111001.csv"
# 假设你的数据是如下格式 CSV，读入DataFrame
# sample_index,ecg,quality,pred,true_label
df = pd.read_csv(filePath)

#==================== 1. 多数投票平滑 ====================
# 假设 pred 列是分类结果（整型类标）
# 设置一个窗口大小 N，对相邻N个样本的预测结果进行多数投票
def majority_voting_smoothing(predictions, window_size=5):
    smoothed = []
    half_win = window_size // 2
    for i in range(len(predictions)):
        start = max(0, i - half_win)
        end = min(len(predictions), i + half_win + 1)
        window = predictions[start:end]
        # 多数投票
        vals, counts = np.unique(window, return_counts=True)
        vote_result = vals[np.argmax(counts)]
        smoothed.append(vote_result)
    return smoothed

df['pred_smoothed_vote'] = majority_voting_smoothing(df['quality'].values, window_size=5)

#==================== 2. 基于阈值的事件触发 ====================
# 这里假设我们有一列概率得分，如 `pred_prob`，范围0~1，越高越说明该窗口为异常事件
# # 如果你目前没有概率分数，可以在你的模型输出中添加该列，或者这里先模拟一下。
if 'pred_prob' not in df.columns:
    # 模拟生成一些概率数据，实际中应来自模型输出
    np.random.seed(42)
    df['pred_prob'] = np.random.rand(len(df))

# 当概率超过阈值时判断为阳性预测，否则为阴性
threshold = 0.7
df['pred_thresh'] = (df['pred_prob'] > threshold).astype(int)

#==================== 3. 事件级合并与过滤 ====================
# 对于基于阈值的事件，我们把连续的1看做一个事件
def find_events(sequence, sample_indices, min_event_length=3, merge_gap=1):
    """
    根据二进制序列sequence(如pred_thresh)识别事件。
    - min_event_length: 事件最短窗口数，短于此长度的事件被过滤掉
    - merge_gap: 如果两个事件之间的空隙小于或等于此值，将它们合并为一个事件
    返回事件列表，每个事件是一个字典，包含起始索引、结束索引、持续窗口数和对应的sample_index范围。
    """
    events = []
    in_event = False
    start_idx = None
    for i, val in enumerate(sequence):
        if val == 1 and not in_event:
            # 事件开始
            in_event = True
            start_idx = i
        elif val == 0 and in_event:
            # 事件结束
            end_idx = i - 1
            length = end_idx - start_idx + 1
            if length >= min_event_length:
                events.append({
                    'start': start_idx, 
                    'end': end_idx, 
                    'length': length,
                    'start_sample': sample_indices[start_idx],
                    'end_sample': sample_indices[end_idx]
                })
            in_event = False
    # 若最后仍在事件中闭合
    if in_event:
        end_idx = len(sequence) - 1
        length = end_idx - start_idx + 1
        if length >= min_event_length:
            events.append({
                'start': start_idx, 
                'end': end_idx, 
                'length': length,
                'start_sample': sample_indices[start_idx],
                'end_sample': sample_indices[end_idx]
            })

    # 合并过近事件
    merged_events = []
    if events:
        events = sorted(events, key=lambda x: x['start'])
        current_event = events[0]
        for e in events[1:]:
            if e['start'] - current_event['end'] <= merge_gap:
                # 合并事件
                current_event['end'] = e['end']
                current_event['end_sample'] = e['end_sample']
                current_event['length'] = current_event['end'] - current_event['start'] + 1
            else:
                merged_events.append(current_event)
                current_event = e
        merged_events.append(current_event)

    return merged_events

events = find_events(df['pred_thresh'].values, df['sample_index'].values, min_event_length=3, merge_gap=2)

print("Detected Events:")
# print(len(events))
for ev in events:
    print(ev)

#==================== 使用事件级结果 ====================
# 根据识别到的事件，可以计算事件级别的TP/FP/FN等指标。
# 假设 true_label 也是二进制（0或1表示没有事件，有事件），
# 可以从真实标注中找到真实事件，然后与预测事件进行IOU或起止点比较，评估事件级指标。

# 示例：假设true_label中同理地提取真实事件
true_events = find_events(df['label'].values, df['sample_index'].values, min_event_length=3, merge_gap=2)

print("True Events:")
for tev in true_events:
    print(tev)

# 之后可以比较 events 和 true_events 进行事件级匹配和统计
# 简单的方法是检查预测事件和真实事件的时间重叠情况

def event_overlap(e1, e2):
    # 简单计算事件重叠长度
    start = max(e1['start_sample'], e2['start_sample'])
    end = min(e1['end_sample'], e2['end_sample'])
    return max(0, end - start + 1)

# 示例：计算简单的TP: 只要有任意重叠即认为TP
tp = 0
for pe in events:
    for te in true_events:
        if event_overlap(pe, te) > 0:
            tp += 1
            break

fp = len(events) - tp
fn = len(true_events) - tp
# 假设我们用一个简单的贪心策略，把预测事件和真实事件做一对一匹配
tp = 0
matched_true = set()  # 记录已经匹配过的真实事件索引
for pe in events:
    best_overlap = 0
    best_idx = None
    for i, te in enumerate(true_events):
        if i not in matched_true:
            overlap = event_overlap(pe, te)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
    if best_idx is not None and best_overlap > 0:
        # 找到与pe最匹配的te，记录为已匹配
        matched_true.add(best_idx)
        tp += 1

fp = len(events) - tp
fn = len(true_events) - tp
print("Event-level TP:", tp)
print("Event-level FP:", fp)
print("Event-level FN:", fn)

# 上述代码只是一个基础的示例框架，你可以根据实际情况（如模型输出是概率还是类别，
# 是否有明确的事件定义等）进行调整和优化。
