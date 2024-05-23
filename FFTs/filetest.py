import os 
import shutil
import pandas as pd

orig_directory_path = './29th_June_EEG_Data'
gen_directory_path = './NewData'

directories = [orig_directory_path, gen_directory_path]

for root, dirs, files in os.walk(orig_directory_path):
    for name in files:
        file_path = os.path.join(root, name)
        if os.path.isfile(file_path):
            shutil.move(file_path, orig_directory_path)

for root, dirs, files in os.walk(orig_directory_path, topdown=False):
    for name in dirs:
        dirpath = os.path.join(root, name)
        shutil.rmtree(dirpath)

for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(directory, filename)
            df = pd.read_excel(file_path)
            print(f'Directory: {directory} | Filename: {filename} | Shape: {df.shape}')