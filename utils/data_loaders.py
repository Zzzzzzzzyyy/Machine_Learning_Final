import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


def encode_date(data):
    data['dteday'] = pd.to_datetime(data['dteday'], format='%Y/%m/%d')
    year = data['dteday'].dt.year
    month = data['dteday'].dt.month
    day = data['dteday'].dt.day
    weekday = data['dteday'].dt.weekday
    month_sin = np.sin(2 * np.pi * (month - 1) / 12)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    data['year'] = year
    data['month'] = month
    data['day'] = day
    data['weekday'] = weekday
    data['month_sin'] = month_sin
    data['month_cos'] = month_cos
    data['weekday_sin'] = weekday_sin
    data['weekday_cos'] = weekday_cos

    data.drop(columns=['dteday'], inplace=True)

    return data


def create_sequences(data, target, target_scaled, features, input_length=96, output_length=96):
    sequences = []
    labels = []
    labels_scaled = []

    for i in range(input_length, len(data) - output_length):
        seq = data[i - input_length:i][features].values
        label = data.iloc[i + 1:i + output_length + 1][target].values
        label_scaled = data.iloc[i + 1:i + output_length + 1][target_scaled].values
        sequences.append(seq)
        labels.append(label)
        labels_scaled.append(label_scaled)

    return np.array(sequences), np.array(labels), np.array(labels_scaled)


def load_data(data_file_path, input_length=96, output_length=96):
    data = pd.read_csv(data_file_path)
    data = encode_date(data)
    # features = ['instant', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday','workingday','weathersit', 'temp',
    #             'atemp', 'hum', 'windspeed', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']
    features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp',
                'atemp', 'hum', 'windspeed']
    target = 'cnt'

    scaler = MinMaxScaler()
    data['cnt_scaled'] = scaler.fit_transform(data['cnt'].values.reshape(-1, 1))

    x, y, y_scaled = create_sequences(data, 'cnt', 'cnt_scaled', features, input_length, output_length)

    x = torch.tensor(x, dtype=torch.float32).cuda()
    y = torch.tensor(y, dtype=torch.float32).cuda()
    y_scaled = torch.tensor(y_scaled, dtype=torch.float32).cuda()

    return x, y, y_scaled, scaler


class BicycleDataset(Dataset):
    def __init__(self, x, y, y_scaled):
        self.x = x
        self.y = y
        self.y_scaled = y_scaled

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.y_scaled[idx]


if __name__ == "__main__":
    train_file_path = '/home/zzy/repository/machine_learning_final/datasets/train_data.csv'
    dataset = BicycleDataset(train_file_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        features = batch['feature']
        targets = batch['target']
        print(features)
        print(targets)
        print(features.shape, targets.shape)