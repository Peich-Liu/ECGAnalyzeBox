import pandas as pd

# Load the CSV file into a DataFrame
file_path = r'c:\Document\sc2024\filtered_ecg_with_qualitynewwithoutFilter.csv'
df = pd.read_csv(file_path)

# Find the consecutive segments where the 'quality' column is 'good'
good_intervals = []
start_index = None

for idx, row in df.iterrows():
    if row['quality'] == 'Good':
        if start_index is None:
            start_index = idx
    else:
        if start_index is not None:
            good_intervals.append([start_index, idx - 1])
            start_index = None

# Add the last segment if it ends with 'good'
if start_index is not None:
    good_intervals.append([start_index, len(df) - 1])

# Output the consecutive good intervals
print(good_intervals)


label_file_path = r'c:\Document\sc2024\data\testData\100001_ANN.csv'
labelDf = pd.read_csv(label_file_path)

# 选择第10, 11, 12列作为 '开始', '结束', 'label'
labelDf = labelDf.iloc[:, [9, 10, 11]]
labelDf.columns = ['start', 'end', 'label']

print(labelDf)

# 创建一个字典，用于存储每个label的区间信息
good_label_intervals = []
bad_label_intervals = []


# 遍历数据帧的每一行，提取开始和结束信息，并按label存储
for _, row in labelDf.iterrows():
    label = row['label']
    start = row['start']
    end = row['end']

    if label == 1.0:
        good_label_intervals.append([start,end])
    elif label == 2.0:
        bad_label_intervals.append([start,end])
print(good_label_intervals)