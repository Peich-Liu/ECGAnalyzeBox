# ECGSignalAnalyzeBox
## Branch Intro
The **Main** branch is the gui code, the **AlgorithmBranch** is the algorithm code.
## Environment
Use this code to create your environment:
```
conda env create -f environment.yaml --name YOUROWNNAME
```
## Test Dataset
A dataset for Brno University of Technology ECG Quality Database as a test, which is the .dat with a .art format. But this dataset does not separate the whole signal into different files.
```
https://physionet.org/content/butqdb/1.0.0/100001/#files-panel
```
## The tasks
1. an analysis of arterial pressures
2. an analysis of ECG focused on Heart Rate Variability
3. BP and HR should also enable a calculation of baroreceptor sensitivity
