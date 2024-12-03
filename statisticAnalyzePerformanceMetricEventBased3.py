import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Load your first CSV file
file_path = r'/Users/liu/Documents/SC2024fall/filtered_ecg_with_qualitynewwithoutFilter.csv'

df = pd.read_csv(file_path)

# Process the first DataFrame to get good intervals
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
if start_index is not None:
    good_intervals.append([start_index, len(df) - 1])

# print("good_intervals",good_intervals)
print("good_intervals",len(good_intervals))


# Load your second CSV file
label_file_path = r'c:\Document\sc2024\data\testData\100001_ANN.csv'
labelDf = pd.read_csv(label_file_path)
labelDf = labelDf.iloc[:, [9, 10, 11]]
labelDf.columns = ['start', 'end', 'label']

# Process the second DataFrame to get good label intervals
good_label_intervals = []
bad_label_intervals = []
for _, row in labelDf.iterrows():
    label = row['label']
    start = row['start']
    end = row['end']
    if label == 1.0:
        good_label_intervals.append([int(start), int(end)])
    elif label == 2.0:
        bad_label_intervals.append([int(start), int(end)])
# print("good_label_intervals",good_label_intervals)

# Ensure that the indices correspond between the two datasets
total_length = len(df)
predicted_labels = np.zeros(total_length, dtype=int)
for interval in good_intervals:
    start_idx, end_idx = interval
    predicted_labels[start_idx:end_idx + 1] = 1

true_labels = np.zeros(total_length, dtype=int)
for interval in good_label_intervals:
    start_idx, end_idx = interval
    # Adjust indices if necessary
    true_labels[start_idx:end_idx + 1] = 1

# Compute the confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

print("Confusion Matrix:")
print(confusion)


def intervals_near(interval1, interval2, tolerance=1000):
    return (interval1[0] - tolerance) <= interval2[1] and (interval2[0] - tolerance) <= interval1[1]

TP = 0
FP = 0
FN = 0

for pred_interval in good_intervals:
    match = False
    for true_interval in good_label_intervals:
        if intervals_near(pred_interval, true_interval, tolerance=5):  # Adjust tolerance as needed
            match = True
            break
    if match:
        TP += 1
    else:
        FP += 1

for true_interval in good_label_intervals:
    match = False
    for pred_interval in good_intervals:
        if intervals_near(pred_interval, true_interval, tolerance=5):
            match = True
            break
    if not match:
        FN += 1

print(f"TP: {TP}, FP: {FP}, FN: {FN}")

