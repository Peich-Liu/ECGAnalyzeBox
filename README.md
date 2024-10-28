# ECGSignalAnalyzeBox
**DO NOT** store the data in this repo
## Update 28.10
The oldMain works well. It is a only ECG framework.
## Environment
Use this code to create your environment:
```
conda env create -f environment.yaml --name YOUROWNNAME
```
## Test Dataset
A dataset for MIT-BIH as a test, which is the .dat with a .art format. But this dataset does not separate the whole signal into different files.
```
https://physionet.org/content/mitdb/1.0.0/
```
## The tasks
1. an analysis of arterial pressures
2. an analysis of ECG focused on Heart Rate Variability
3. BP and HR should also enable a calculation of baroreceptor sensitivity
