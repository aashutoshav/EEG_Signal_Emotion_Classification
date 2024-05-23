import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
 
folder_path = ('./Datasets/29th_June_EEG_Data')
for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    if not os.path.isdir(item_path):
         continue
    for filename in os.listdir(item_path):
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                writer = pd.ExcelWriter("./NewData/"+filename, engine = 'xlsxwriter')
                file_path = os.path.join(item_path, filename)
                xls = pd.ExcelFile(file_path)
                for sheet_name in xls.sheet_names:
                    df_original_data = pd.read_excel(xls, sheet_name=sheet_name, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], skiprows = [1])
                    for column_name in df_original_data.columns:
                        eeg_data = np.array(df_original_data[column_name])
    
                        # Perform Fourier Transform
                        signal_fft = np.fft.rfft(eeg_data)
                        random_noise=np.random.uniform(-2,2,signal_fft.shape) #0 to 2 scaling of amplitude
                        signal_fft = signal_fft*random_noise
 
                        # Perform Inverse Fourier Transform
                        new_signal = np.fft.irfft(signal_fft)
 
                        df_original_data[column_name]=pd.Series(new_signal)
                    
                new_row = {'Time':'(HH-MM-SS)','FP2 - REF':'uV','F4 - REF':'uV','C4 - REF':'uV','P4 - REF':'uV','FP1 - REF':'uV','F3 - REF':'uV',
                        'C3 - REF':'uV','P3 - REF':'uV','F8 - REF':'uV','T4 - REF':'uV','T6 - REF':'uV','O2 - REF':'uV','F7 - REF':'uV',
                        'T3 - REF':'uV','T5 - REF':'uV','O1 - REF':'uV','CZ - REF':'uV','PZ - REF':'uV','FZ - REF':'uV','EMG':'uV','EKG':'uV'}
                df_original_data.loc[0] = new_row
                df_original_data.to_excel(writer, sheet_name = sheet_name)
                writer.close()
                print("Data saved to /NewData/"+filename)
                