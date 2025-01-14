import torch
import torch.nn as nn


def calculate_metrics(gt, pred):
    mse = nn.MSELoss()(pred, gt).item()
    mae = torch.mean(torch.abs(pred - gt)).item()
    std_dev = torch.std(pred).item()

    return mse, mae, std_dev