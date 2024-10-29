# ECGSignalAnalyzeBox
**DO NOT** store the data in this repo
## Update 29.10
The main.py is updated in this version, which is same with the oldMain.py.
## Environment
Use this code to create your environment:
```
conda env create -f environment.yaml --name YOUROWNNAME
```
## Test Dataset
The new dataset test is the Fantasia Database
```
https://physionet.org/content/fantasia/1.0.0/subset/#files-panel
```
A dataset for MIT-BIH as a test, which is the .dat with a .art format. But this dataset does not separate the whole signal into different files.
```
https://physionet.org/content/mitdb/1.0.0/
```
## The tasks
1. an analysis of arterial pressures
2. an analysis of ECG focused on Heart Rate Variability
3. BP and HR should also enable a calculation of baroreceptor sensitivity
