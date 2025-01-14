import torch
from utils.data_loaders import load_data, BicycleDataset
from torch.utils.data import DataLoader
from models.LSTM import LSTMModel
from utils.average_meter import calculate_metrics
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from configs import get_configs
from utils.plot import plot
from models.Improved_model import ImprovedModel
from models.Transformer import TransformerModel


def test_net(model=None):
    configs = get_configs()
    data_file_path = './datasets/test_data.csv'
    input, gt, gt_scaled, scaler = load_data(data_file_path, input_length=configs['input_length'], output_length=configs['output_length'])
    feature_length = input.shape[2]

    test_dataset = BicycleDataset(input, gt, gt_scaled)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model = LSTMModel(
    #     input_size=feature_length,
    #     output_length=configs['output_length'],
    #     hidden_size=configs['hidden_size'],
    #     num_layers=configs['num_layers']
    # )

    # model = TransformerModel(
    #     input_size=feature_length,
    #     output_length=configs['output_length'],
    #     hidden_size=configs['hidden_size'],
    #     num_layers=configs['num_layers'],
    #     num_heads=configs['num_head']
    # )

    if not model:
        model = ImprovedModel(
            input_size=feature_length,
            output_length=configs['output_length'],
            hidden_size=configs['hidden_size'],
            num_layers=configs['num_layers']
        )

    model = model.to('cuda')
    model.load_state_dict(torch.load('./result/Improved_model_240/ckpt.pth'))

    model.eval()

    criterion = nn.MSELoss()

    total_gt = []
    total_pred = []
    count = 0

    with torch.no_grad():
        running_loss = 0.0
        num_batches = len(test_dataloader)

        with tqdm(test_dataloader) as t:
            for idx, data in enumerate(t):
                inputs = data[0]
                targets = data[1]
                targets_scaled = data[2]

                outputs = model(inputs)
                outputs = outputs.cpu()
                outputs = scaler.inverse_transform(outputs)

                outputs = torch.tensor(outputs, dtype=torch.float32, device='cuda:0')
                loss = criterion(outputs, targets)
                running_loss += loss.item()

                total_gt.append(targets)
                total_pred.append(outputs)

                t.set_description(f'Test - Batch {idx + 1}/{num_batches}')
                t.set_postfix(loss='%s' % ['%.4f' % loss])

                if count > 300:
                    plot(outputs.squeeze(), targets.squeeze())

                count += 1

        total_gt = torch.cat(total_gt, dim=0)
        total_pred = torch.cat(total_pred, dim=0)
        mse, mae, std_dev = calculate_metrics(total_gt, total_pred)
        print(f'Test Loss: {running_loss / num_batches:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Std_Dev: {std_dev:.4f}')


if __name__ == '__main__':
    test_net()
