import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm


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


class LSTMModel(nn.Module):
    def __init__(self, num_rows, num_columns, bidirectional_lstm=False):
        super().__init__()
        self.num_columns = num_columns
        self.lstm_layer = nn.ModuleList(
            [
             	nn.LSTM(
                    input_size=num_rows,
                    hidden_size=128,
                    bidirectional=bidirectional_lstm,
                    batch_first=True,
                )
                for i in range(self.num_columns)
            ]
	    )
        lstm_output_size = 128 * 2 if bidirectional_lstm else 128
        self.fc1 = nn.Linear(lstm_output_size * self.num_columns, 256)
        self.fc2 = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = []
        for i in range(self.num_columns):
            t = x[:, :, i]
            t, _ = self.lstm_layer[i](t)
            out.append(t)
        out = torch.cat(out, dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(5120, 19, bidirectional_lstm=True)

trainloader, testloader = EEGLoaders(
    data_pth, json_pth, nrows_per_item, batch_size, train_split=0.7
)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    params=model.parameters(),
    lr=lr,
)

for epoch in range(num_epochs):
    model.train()
    training_loss = 0.0
    training_acc = 0.0
    for i, (data, labels) in tqdm(
        enumerate(trainloader), desc=f"Training Epoch {epoch+1}/{num_epochs}"
    ):
        outputs = model(data.to(device))
        optimizer.zero_grad()
        loss = criterion(outputs, labels.to(device))
        training_loss += loss.item()
        acc = torch.sum(torch.argmax(outputs, -1).cpu() == torch.argmax(labels, 1).cpu())
        training_acc += acc.item()
        loss.backward()
        optimizer.step()
        del loss
    training_loss /= float(len(trainloader.dataset))
    training_acc /= float(len(trainloader.dataset))
    print(
	f"Training Epoch {epoch+1}/{num_epochs}: Loss = {training_loss:.6f}, Accuracy = {training_loss:.6f}")
    
    model.eval()
    testing_loss = 0.0
    testing_acc = 0.0
    with torch.inference_mode():
        for i, (data, labels) in tqdm(
            enumerate(testloader), desc=f"Testing Epoch {epoch+1}/{num_epochs}"
        ):
            outputs = model(data.to(device))
            loss = criterion(outputs, labels.to(device))
            testing_loss += loss.item()
            acc = torch.sum(torch.argmax(outputs, -1).cpu() == torch.argmax(labels, 1).cpu())
            testing_acc += acc.item()
            del loss
        testing_loss /= float(len(testloader.dataset))
        testing_acc /= float(len(testloader.dataset))
        print(f"Testing Epoch {epoch+1}/{num_epochs}: Loss = {testing_loss:.6f}, Accuracy = {testing_acc:.6f}")
