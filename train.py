import torch
from utils.data_loaders import load_data, BicycleDataset
from torch.utils.data import DataLoader
from models.LSTM import LSTMModel
from utils.average_meter import calculate_metrics
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from configs import get_configs
from models.Improved_model import ImprovedModel
from torch.optim.lr_scheduler import StepLR
from test import test_net


def train_net():
    configs = get_configs()
    data_file_path = './datasets/train_data.csv'
    input, gt, gt_scaled, scaler = load_data(data_file_path, input_length=configs['input_length'], output_length=configs['output_length'])
    feature_length = input.shape[2]

    train_dataset = BicycleDataset(input, gt, gt_scaled)
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)

    # model = LSTMModel(
    #     input_size=feature_length,
    #     output_length=configs['output_length'],
    #     hidden_size=configs['hidden_size'],
    #     num_layers=configs['num_layers']
    # )

    model = ImprovedModel(
        input_size=feature_length,
        output_length=configs['output_length'],
        hidden_size=configs['hidden_size'],
        num_layers=configs['num_layers']
    )

    model = model.to('cuda')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs['learning_rate'])
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)

    num_epochs = configs['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_gt = []
        total_pred = []
        num_batches = len(train_dataloader)

        with tqdm(train_dataloader) as t:
            for idx, data in enumerate(t):
                inputs = data[0]
                targets = data[1]
                targets_scaled = data[2]

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets_scaled)
                loss.backward()
                optimizer.step()

                total_gt.append(targets_scaled)
                total_pred.append(outputs.squeeze())
                running_loss += loss.item()

                lr = optimizer.param_groups[0]['lr']
                t.set_description(f'Epoch[{epoch+1}/{num_epochs}][Batch{idx+1}/{num_batches}]')
                t.set_postfix(loss='%s' % ['%.4f' % loss], lr='%.6f' % lr)

        total_gt = torch.cat(total_gt, dim=0)
        total_pred = torch.cat(total_pred, dim=0)
        mse, mae, std_dev = calculate_metrics(total_gt, total_pred)

        tqdm.write(f'Epoch[{epoch+1}/{num_epochs}] : Loss:{running_loss/len(train_dataloader):.4f}, MSE:{mse:.4f}, MAE:{mae:.4f}, Std_Dev:{std_dev:.4f}')
        scheduler.step()
        test_net(model)
        torch.save(model.state_dict(), f'./result/ckpt-epoch-{epoch+1}.pth')

    # torch.save(model.state_dict(), './result/ckpt.pth')


if __name__ == '__main__':
    train_net()
