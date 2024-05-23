import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
 
print('start')
# for original data
folder_path_original = ('./29th_June_EEG_Data')
for item in os.listdir(folder_path_original):
    item_path = os.path.join(folder_path_original, item)
    print('original')
    if os.path.isdir(item_path):
        print("Folder:", item_path)
    if not os.path.isdir(item_path):
         continue
    big_pos_org=[]
    big_neg_org=[]
    big_neutral_org=[]
    for filename in os.listdir(item_path):
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                # Construct the full file path
                file_path = os.path.join(item_path, filename)
                print("File:", file_path) 
                xls = pd.ExcelFile(file_path)
                for sheet_name in xls.sheet_names:
                    #print("Sheet name:", sheet_name)
                    # Read the current sheet into a DataFrame
                    df_original_data = pd.read_excel(xls, sheet_name=sheet_name, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], skiprows = [1])
                    #df = pd.read_excel(excel_file, usecols=lambda name: name != 'Column1')
                    #print(df_original_data)
                    nu=df_original_data.to_numpy()
                    #print(nu)
                    if "Pos" in filename:
                        big_pos_org.append(nu)
                    if "Neg" in filename:
                        big_neg_org.append(nu)
                    if "Neutral" in filename:
                        big_neutral_org.append(nu)
 
# for generated data                       
folder_path_original = ('./NewData')
'''
for item in os.listdir(folder_path_original):
    item_path = os.path.join(folder_path_original, item)
    if os.path.isdir(item_path):
        print("Folder:", item_path)
    if not os.path.isdir(item_path):
         continue
'''
big_pos_gen=[]
big_neg_gen=[]
big_neutral_gen=[]
for filename in os.listdir(item_path):
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        # Construct the full file path
        file_path = os.path.join(item_path, filename)
        print("File:", file_path) 
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            #print("Sheet name:", sheet_name)
            # Read the current sheet into a DataFrame
            df_original_data = pd.read_excel(xls, sheet_name=sheet_name, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], skiprows = [1])
            #df = pd.read_excel(excel_file, usecols=lambda name: name != 'Column1')
            #print(df_original_data)
            nu=df_original_data.to_numpy()
            #print(nu)
            if "Pos" in filename:
                big_pos_gen.append(nu)
            if "Neg" in filename:
                big_neg_gen.append(nu)
            if "Neutral" in filename:
                big_neutral_gen.append(nu)
 
# making groups
# big_neg_gen=np.concatenate(big_neg_gen, axis=0)
# big_neg_org=np.concatenate(big_neg_org,axis=0)
# big_neutral_gen=np.concatenate(big_neutral_gen,axis=0)
# big_neutral_org=np.concatenate(big_neutral_org,axis=0)
# big_pos_gen=np.concatenate(big_pos_gen,axis=0)
# big_pos_org=np.concatenate(big_pos_org,axis=0)    
 
big_original = np.concatenate([big_pos_org, big_neg_org, big_neutral_org], axis=0)
big_generated = np.concatenate([big_neg_gen,big_pos_org,big_neutral_gen])
 
print('Shapes:',big_original.shape, big_generated.shape)
num_classes = 2
 
big_original_tensor = torch.from_numpy(big_original)
big_generated_tensor = torch.from_numpy(big_generated)
#reduce dimensionality
train_tnse_2 = torch.flatten(big_original_tensor, 1,2)
test_tsne_2 = torch.flatten(big_generated_tensor,1,2)
def test_tensor_batch():
    return torch.rand([8,num_classes])
 
def train_tensor_batch():
    return torch.rand([8,num_classes])
 
tsne = TSNE(n_components=2,random_state=42) # 2D tensor as result
 
 
train_tsne = tsne.fit_transform(train_tnse_2)
test_tsne = tsne.fit_transform(test_tsne_2)
 
 
 
print('Shapes:', train_tsne.shape, test_tsne.shape)
np.save('original',train_tsne)
np.save('generated',test_tsne)
 
# Step 5: Visualize t-SNE plots
plt.figure(figsize=(10, 5))
 
# Train t-SNE plot
plt.subplot(1, 2, 1)
plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c='r')
plt.title('Original')
 
# Test t-SNE plot
plt.subplot(1, 2, 2)
plt.scatter(test_tsne[:, 0], test_tsne[:, 1], c='b')
plt.title('Generated')
 
plt.tight_layout()
plt.subplot(1, 2, 1)
plt.savefig('original.png')
plt.subplot(1, 2, 2)
plt.savefig('generated.png')

combined_tsne = np.concatenate((train_tsne, test_tsne), axis=0)
labels = np.array(['Original'] * train_tsne.shape[0] + ['Generated'] * test_tsne.shape[0])

plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    if label == 'Original':
        plt.scatter(combined_tsne[i, 0], combined_tsne[i, 1], c='r', label=label)
    else:
        plt.scatter(combined_tsne[i, 0], combined_tsne[i, 1], c='b', label=label)
        
plt.title('Original vs. Generated Data')
# plt.legend()
plt.tight_layout()

output_dir = '/scratch/abhijitdas/EEG'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
combined_plot_path = os.path.join(output_dir, 'combined_plot.png')
plt.savefig(combined_plot_path)
# plt.show()
# plt.savefig('trial2.png')