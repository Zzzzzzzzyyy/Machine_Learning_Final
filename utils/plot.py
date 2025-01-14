import numpy as np
import matplotlib.pyplot as plt


def plot(pred, targets):
    pred = pred.cpu()
    targets = targets.cpu()
    time_steps = np.arange(len(pred))
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, pred, label='pred', color='red', linestyle='-', linewidth=2)
    plt.plot(time_steps, targets, label='gt', color='blue', linestyle='-', linewidth=2)
    plt.title('Predict')
    plt.xlabel('Time Steps')
    plt.ylabel('cnt')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
