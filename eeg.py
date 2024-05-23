import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm

# -------SPECIFICS---------

data_pth = "."  # EEG directory
json_pth = "./rowstog.json"

# -----HYPERPARAMETERS-----

nrows_per_item = 64
lr = 1e-4
batch_size = 128
num_epochs = 10

data_dict = {}

with open(json_pth) as f:
    ref = json.load(f)

for filepath in tqdm(ref, desc="Reading Data..."):
    fullpth = os.path.join(data_pth, filepath)
    excel = pd.read_excel(fullpth, sheet_name=None, engine="openpyxl")
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
        # print(data.shape)
        # data shape [bs,ft,1] ft=19
        new_data = []
        for x in range(0, 16):
            temp = data[:, x : x + 4]  # window size is 4
            new_data.append(temp)
        new_data = torch.stack(new_data)
        # new_data=torch.tensor(new_data)
        # print(new_data.shape)
        data = torch.reshape(new_data, (-1, 16, 4))  # window size is 4
        # data shape [bs.ft,4,1] ft=19
        label = self.whatsthelabel(filepath)
        return data, label


def EEGLoaders(
    data_path, json_path, nrows_per_item, batch_size, train_split, num_workers=0
):
    eeg_set = EEGSignals(
        data_path=data_path, json_path=json_path, nrows_per_item=nrows_per_item
    )
    size = len(eeg_set)
    train_size = int(train_split * size)
    test_size = size - train_size
    train_dataset, test_dataset = random_split(eeg_set, [train_size, test_size])
    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


class SlitCNN_extra_cols(nn.Module):

    kernel_size_1 = (8, 1)
    kernel_size_2 = (4, 1)
    stride = 1

    def __init__(self, num_rows, num_columns, num_classes, window_size=4):
        super().__init__()

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_classes = num_classes
        self.window_size = window_size

        self.conv1 = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.window_size,
                    out_channels=64,
                    kernel_size=self.kernel_size_1,
                    stride=self.stride,
                )
                for i in range(self.num_columns)
            ]
        )

        self.conv2 = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=self.kernel_size_2,
                    stride=self.stride,
                )
                for i in range(self.num_columns)
            ]
        )

        self.norm = nn.LayerNorm(
            [
                64,
                (self.num_rows - self.kernel_size_1[0] + self.stride) // self.stride,
                1,
            ]
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1))
        self.lrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)

        self.fc1 = nn.ModuleList(
            [
                nn.Linear(
                    in_features=128
                    * int(
                        (
                            self.num_rows
                            - self.kernel_size_1[0]
                            + self.stride
                            - self.kernel_size_2[0]
                            + self.stride
                        )
                        // (2 * self.stride)
                    ),
                    out_features=128,
                )
                for i in range(self.num_columns)
            ]
        )

        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(
            in_features=128 * self.num_columns, out_features=self.num_classes
        )

    def forward(self, x):
        to_cat = []

        for i in range(self.num_columns):
            t = x[:, :, i]
            t = torch.reshape(t, (-1, self.window_size, t.shape[1], 1))
            print(t.shape)
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
        out = torch.cat(to_cat, dim=1)
        out = self.fc2(out)
        return out


model = SlitCNN_extra_cols(64, 19 - 4 + 1, 3, 4)  # 4 here is the window size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr)

trainloader, testloader = EEGLoaders(
    data_path=data_pth,
    json_path=json_pth,
    nrows_per_item=nrows_per_item,
    batch_size=batch_size,
    train_split=0.75,
)

for epoch in range(num_epochs):
    model.train()
    training_loss = 0.0
    training_acc = 0.0
    for i, (data, labels) in tqdm(
        enumerate(trainloader), desc=f"Training Epoch {epoch+1}/{num_epochs}"
    ):
        print(data.shape)
        outputs = model(data.to(device))
        optimizer.zero_grad()
        loss = criterion(outputs, labels.to(device))
        training_loss += loss.item()
        acc = torch.sum(
            torch.argmax(outputs, -1).cpu() == torch.argmax(labels, 1).cpu()
        )
        training_acc += acc.item()
        loss.backward()
        optimizer.step()
        del loss
    training_loss /= float(len(trainloader.dataset))
    training_acc /= float(len(trainloader.dataset))
    print(
        f"Training Epoch {epoch+1}/{num_epochs}: Loss = {training_loss:.6f}, Accuracy = {training_loss:.6f}"
    )

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
            acc = torch.sum(
                torch.argmax(outputs, -1).cpu() == torch.argmax(labels, 1).cpu()
            )
            testing_acc += acc.item()
            del loss
        testing_loss /= float(len(testloader.dataset))
        testing_acc /= float(len(testloader.dataset))
        print(
            f"Testing Epoch {epoch+1}/{num_epochs}: Loss = {testing_loss:.6f}, Accuracy = {testing_acc:.6f}"
        )
