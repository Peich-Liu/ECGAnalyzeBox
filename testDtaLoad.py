import pyreadstat
import pandas as pd
import wfdb



file_path = r'C:/Users/lpcn5/Downloads/1.1'

record = wfdb.rdrecord(file_path)

print(record)


