import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from torch.nn import Sequential, Linear, ReLU


# -------SPECIFICS---------

data_pth = "."
json_pth = "./rows.json"

# -----HYPERPARAMETERS-----

nrows_per_item = 5120
lr = 1e-4
batch_size = 8
num_epochs = 10

data_dict = {}

with open(json_pth) as f:
    ref = json.load(f)
    
for filepath in tqdm(ref, desc='Reading Data...'):
    fullpth = os.path.join(data_pth,filepath)
    excel = pd.read_excel(fullpth, sheet_name=None)
    num_sheets = len(excel.items())
    full_df = 0
    for i in range(1, 1 + num_sheets):
        df = pd.read_excel(
            filepath,
            sheet_name=f"Pg-{i}",
            index_col=None,
            usecols=list(range(1, 1 + 19)),
            skiprows=[1],
        )
        if i == 1:
            full_df = df
        else:
            full_df = pd.concat([full_df, df])
    data_dict[filepath] = full_df

class EEGSignals(Dataset):
    def __init__(
        self,
	data_path,  # PATH TO DIRECTORY CONTAINING "29th_June_EEG_Data"
        json_path,  # EXACT PATH OF "rows.json"
        nrows_per_item,  # NUMBER OF ROWS CONSIDERED AS A SINGLE DATA ITEM. IGNORES LEFTOVERS AT THE END
    ):
        with open(json_path) as f:
            ref = json.load(f)
        self.paths, self.nrows = zip(*ref.items())
        self.num_items = sum([n_rows // nrows_per_item for n_rows in self.nrows])
        self.erows = [n_row - (n_row % nrows_per_item) for n_row in self.nrows]
        self.crows = [sum(self.erows[: i + 1]) for i in range(len(self.erows))]
        self.nrows_per_item = nrows_per_item
        self.data_path = data_path
        
    def whatsthelabel(self, filepath):
        filename = filepath.split("/")[-1]
        pos = int("Pos" in filename)
        neg = int("Neg" in filename)
        neu = int("Neutral" in filename)
        assert (
            pos + neg + neu == 1
        ), f'Invalid Filename "{filename}".\nFilename must contain exactly one of the strings "Pos", "Neg" or "Neutral".'
        return torch.tensor([pos, neu, neg], dtype=torch.float32)

    def __len__(self):
        return self.num_items
    
    def __getitem__(self, idx):
        searchable = idx * self.nrows_per_item
        j, s = 0, 0
        for i, d in enumerate(self.crows):
            if searchable < d:
                j = i
                searchable -= s
                break
            s = d
        filepath = self.paths[j]
        df = data_dict[filepath]
        data = df[searchable : searchable + self.nrows_per_item]
        data = torch.from_numpy(data.values).to(torch.float32)
        label = self.whatsthelabel(filepath)
        return data, label
    
    
def EEGLoaders(
    data_path,  # PATH TO DIRECTORY CONTAINING "29th_June_EEG_Data"
    json_path,  # EXACT PATH OF "rows.json"
    nrows_per_item,  # NUMBER OF ROWS CONSIDERED AS A SINGLE DATA ITEM. IGNORES LEFTOVERS AT THE END
    batch_size,
    train_split,
    num_workers=0,
):
    eeg_set = EEGSignals(data_path, json_path, nrows_per_item)
    size = len(eeg_set)
    train_size = int(train_split * size)
    test_size = size - train_size
    train_dataset, test_dataset = random_split(eeg_set, [train_size, test_size])
    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

class SlitCNN(nn.Module):
    kernel_size_1 = (30, 1)
    kernel_size_2 = (8, 1)
    stride = 1
    
    def __init__(self, num_rows, num_cols, num_classes):
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_classes = num_classes
        
        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=self.kernel_size_1,stride=self.stride)
            for i in range(self.num_cols)
        ])
        
        self.conv2 = nn.ModuleList([
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=self.kernel_size_2,stride=self.stride)
            for i in range(self.num_cols)
        ])

        self.norm = nn.LayerNorm([64,(self.num_rows-self.kernel_size_1[0]+self.stride)//self.stride,1])
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1,end_dim=2)
        
        self.fc1 = nn.ModuleList([
            nn.Linear(in_features=128*int((self.num_rows-self.kernel_size_1[0]+self.stride-self.kernel_size_2[0]+self.stride)//(2*self.stride)),out_features=128)
            for i in range(self.num_cols)
        ])

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=128*self.num_cols, out_features=self.num_cols)
        
    def forward(self,x):
        to_cat = []

        for i in range(self.num_cols):
            t = x[:,:,i:(i+1)]
            t = torch.unsqueeze(t,dim=1)
            t = self.conv1[i](t)
            t = self.lrelu(t)
            t = self.norm(t)
            t = self.conv2[i](t)
            t = self.lrelu(t)
            t = self.maxpool(t)
            t = self.flatten(t)
            t = torch.squeeze(t)
            t = self.fc1[i](t)
            t = self.lrelu(t)
            t = self.dropout(t)
            to_cat.append(t)
        out = torch.cat(to_cat,dim=1)
        out = self.fc2(out)
        return out
    
# Changed the generator class to not have slitcnn because of the dimension issues caused by the line 
#             t = x[:,:,i:(i+1)]
    
# class Generator(nn.Module):
#     def __init__(self, latent_size, class_size, data_dim):
#         super(Generator, self).__init__()
#         self.latent_size = latent_size
#         self.class_size = class_size
#         self.data_dim = data_dim
        
#         self.generator = Sequential(
#             Linear(in_features=101, out_features=256),
#             ReLU(),
#             Linear(in_features=256, out_features=512),
#             ReLU(),
#             Linear(in_features=512, out_features=data_dim)
#         )
        
#         self.slitcnn = SlitCNN(num_rows=5120, num_cols=19, num_classes=2)
        
#     def forward(self, noise, labels):
#         print(f"Shape of noise = {noise.shape}")
#         print(f"Shape of labels = {labels.shape}")
#         noise = self.slitcnn(noise)
#         noise = noise.view(noise.size(0), -1) # Not needed since noise is already flattened, but doesn't hurt to have this line
#         print(f"Shape of noise @ 1 = {noise.shape}")
#         X = torch.cat((noise, labels), dim=1)
#         print(f"Shape of Xnoise @ 2 = {X.shape}")
#         # return self.fc2(self.dropout(self.lrelu(self.fc1(self.flatten(self.maxpool(self.lrelu(self.conv2(self.norm(self.lrelu(self.conv1(X)))))))))))
#         return self.generator(X)

class Generator(nn.Module):
    def __init__(self, latent_size, class_size, data_dim):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.class_size = class_size
        self.data_dim = data_dim

        self.generator = Sequential(
            Linear(in_features=latent_size + class_size - 2, out_features=256),
            ReLU(),
            Linear(in_features=256, out_features=512),
            ReLU(),
            Linear(in_features=512, out_features=data_dim)
        )

    def forward(self, noise, labels):
        X = torch.cat((noise, labels), dim=1)
        return self.generator(X)

class Discriminator(nn.Module):
    def __init__(self, data_dim, class_size):
        super(Discriminator, self).__init__()
        self.data_dim = data_dim
        self.class_size = class_size
        
        self.slitcnn = SlitCNN(num_rows=5120, num_cols=19, num_classes=2)
        
        # THE IDEA HERE IS TO SPLIT THE SHAPE WHICH IS (8, 97283) INTO 4 PARTS EACH OF SHAPE (8, 97283//4), AND THE LAST ONE OF 
        # SHAPE (8, 96283 - 3*(97283)//4)
        # THEN EACH SPLIT OF COLUMNS RUNS THROUGH A SLITCNN AND THE FINAL OUTPUT OF EACH OF THE 4 SPLITS GOES THROUGH A LINEAR LAYER WITH A 
        # SIGMOID ACTIVATION, WHICH COMPENSATES FOR THE OUT_FEATURES HAVING TO BE 1
        
        self.discriminator = Sequential(
            Linear(in_features=22, out_features=256),
            ReLU(),
            Linear(in_features=256, out_features=512),
            ReLU(),
            Linear(in_features=512, out_features=256),
            ReLU(),
            Linear(in_features=256, out_features=1),
            nn.Sigmoid() # Binary Classification so using Sigmoid
        )
        
    def forward(self, data, labels):
        print(f"Shape of data = {data.shape}")
        print(f"Shape of labels = {labels.shape}")
        data = self.slitcnn(data)
        data = data.view(data.size(0), -1)
        print(f"Shape of data @ 1 = {data.shape}")
        X = torch.cat((data, labels), dim=1)
        print(f"Shape of X @ 2 = {X.shape}")
        return self.discriminator(X)
    
class Discriminator2(nn.Module):
    def __init__(self, data_dim, class_size):
        super(Discriminator2, self).__init__()
        self.data_dim = data_dim
        self.class_size = class_size
        
        self.slitcnn = SlitCNN(num_rows=5120, num_cols=19, num_classes=2)

        
        self.discriminator = Sequential(
            Linear(in_features=5121, out_features=512),
            ReLU(),
            Linear(in_features=512, out_features=256),
            ReLU(),
            Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )
        
    def forward(self, data, labels):
        print(f'Shape of data = {data.shape}')
        print(f'Shape of labels = {labels.shape}')
        data = data.view(data.size(0), -1)
        print(f'Shape of data @ 1 = {data.shape}')
        X = torch.cat((data, labels), dim=1)
        print(f'Shape of X @ 2 = {X.shape}')
        return self.discriminator(X)
    
trainloader, testloader = EEGLoaders(data_path=data_pth, json_path=json_pth, nrows_per_item=nrows_per_item, batch_size=batch_size, train_split=0.7)
data, _ = next(iter(trainloader))
latent_size = 100
class_size = 3
data_dim = data.size(1)

generator = Generator(latent_size=latent_size, class_size=class_size, data_dim=data_dim)
discriminator = Discriminator(data_dim=19, class_size=class_size)
discriminator2 = Discriminator2(data_dim=19, class_size=class_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)
discriminator2.to(device)

criterion = nn.CrossEntropyLoss()
gen_optimizer = optim.Adam(params=generator.parameters(), lr=lr)
dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=lr)
dis2_optimizer = optim.Adam(params=discriminator2.parameters(), lr=lr)

for epoch in range(num_epochs):
    print(f'Shape of Input Data: {data.shape}')
    print(f'Batch Size: {batch_size} | Data_Dim: {data_dim}')
    
    generator.train()
    discriminator.train()
    discriminator2.train()
    
    training_loss, training_accuracy = 0.0, 0.0
    
    for i, (data, labels) in tqdm(
        enumerate(trainloader), desc=f'Training Epoch {epoch+1}/{num_epochs}'
    ):
        data, labels = data.to(device), labels.to(device)
        dis_optimizer.zero_grad()
        print(f"Size of data = {data.shape}")        
        real_outputs = discriminator(data, labels)
        real_loss = criterion(real_outputs, torch.ones_like(real_outputs, dtype=torch.float32).to(device))
        
        noise = torch.randn(len(data), latent_size).to(device)
        
        fake_labels = torch.randint(0, class_size, (len(data), 1)).to(device)
        
        fake_data = generator(noise, fake_labels)
        
        fake_outputs = discriminator2(fake_data, fake_labels)
        fake_loss = criterion(fake_outputs, torch.zeros_like(fake_outputs, dtype=torch.float32).to(device))
        
        dis_loss = real_loss + fake_loss
        dis_loss.backward()
        dis_optimizer.step()
        
        gen_optimizer.zero_grad()
        
        noise = torch.randn(len(data), latent_size).to(device)
        
        fake_labels = torch.randint(0, class_size, (len(data), 1)).to(device)
        fake_data = generator(noise, fake_labels)

        fake_outputs = discriminator2(fake_data, fake_labels)
        gen_loss = criterion(fake_outputs, torch.ones_like(fake_outputs, dtype=torch.float32).to(device))

        gen_loss.backward()
        gen_optimizer.step()
        
        batch_loss = dis_loss + gen_loss
        training_loss += batch_loss.item()
        _, predicted_labels = torch.max(fake_outputs.data, 1)
        correct_predictions = (predicted_labels == fake_labels).sum().item()
        training_accuracy += correct_predictions / len(data)
        
    training_loss /= len(trainloader)
    training_accuracy /= len(trainloader)
    print(
	f"Training Epoch {epoch+1}/{num_epochs}: Loss = {training_loss:.6f}, Accuracy = {training_loss:.6f}")

    generator.eval()
    discriminator.eval()
    discriminator2.eval()

    testing_loss, testing_accuracy = 0.0, 0.0

    with torch.inference_mode():
        for i, (data, labels) in tqdm(
            enumerate(testloader), desc=f'Testing Epoch {epoch+1}/{num_epochs}'
        ):
            data, labels = data.to(device), labels.to(device)
            
            real_outputs = discriminator(data, labels)
            real_loss = criterion(real_outputs, torch.ones_like(real_outputs, dtype=torch.float32).to(device))

            noise = torch.randn(len(data), latent_size).to(device)
            fake_labels = torch.randint(0, class_size, (len(data), 1)).to(device)
            fake_data = generator(noise, fake_labels)
            
            fake_outputs = discriminator2(fake_data, fake_labels)
            fake_loss = criterion(fake_outputs, torch.zeros_like(fake_outputs, dtype=torch.float32).to(device))
            
            dis_loss = real_loss + fake_loss
            
            noise = torch.randn(len(data), latent_size).to(device)
            fake_labels = torch.randint(0, class_size, (len(data), 1)).to(device)
            fake_data = generator(noise, fake_labels)

            fake_outputs = discriminator2(fake_data, fake_labels)
            gen_loss = criterion(fake_outputs, torch.ones_like(fake_outputs, dtype=torch.float32).to(device))

            batch_loss = dis_loss + gen_loss
            testing_loss += batch_loss.item()
            _, predicted_labels = torch.max(fake_outputs.data, 1)
            correct_predictions = (predicted_labels == fake_labels).sum().item()
            testing_accuracy += correct_predictions / len(data)
            
        testing_loss /= len(testloader)
        testing_accuracy /= len(testloader)
        print(f'Testing Epoch: {epoch+1}/{num_epochs}: Loss: {testing_loss:.6f}, Accuracy: {testing_accuracy:.6f}')


def generate_and_save_synthetic_data(generator, num_samples, latent_size, class_size, data_dim, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator.to(device)
    generator.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for class_label in range(class_size):
        synthetic_data = []
        synthetic_labels = []
        
        with torch.inference_mode():
            for _ in tqdm(range(num_samples), desc=f"Generating Synthetic Data for Class {class_label}"):
                noise = torch.randn(1, latent_size).to(device)
                labels = torch.tensor([[class_label]], dtype=torch.float32).to(device)
                
                fake_data = generator(noise, labels)
                synthetic_data.append(fake_data[0].cpu().numpy())
                synthetic_labels.append(class_label)
                
        columns = [f"Feature_{i}" for i in range(data_dim)]
        synthetic_data_df = pd.DataFrame(synthetic_data, columns=columns)
        synthetic_labels_df = pd.DataFrame(synthetic_labels, columns=["Class"])
        
        synthetic_data_with_labels = pd.concat([synthetic_data_df, synthetic_labels_df], axis=1)
        
        output_file = os.path.join(output_dir, f"synthetic_class_{class_label}.xlsx")
        synthetic_data_with_labels.to_excel(output_file, index=False)

        print(f"Synthetic data for Class {class_label} saved to {output_file}")

initial_data_size = len(trainloader.dataset)
num_samples_per_class = int(initial_data_size * 0.1)

num_samples = int(num_samples_per_class / class_size)
output_dir = '/scratch/abhijitdas/EEG/_EEG_Data'

generate_and_save_synthetic_data(generator=generator, num_samples=num_samples, latent_size=latent_size, class_size=class_size, data_dim=data_dim, output_dir=output_dir)
