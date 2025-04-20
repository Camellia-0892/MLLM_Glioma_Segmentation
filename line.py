# coding = gbk
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

epochs = np.arange(1, 100)

initial_loss = 1.7
final_loss = 0.2
trend = initial_loss * np.exp(-0.1 * epochs) + final_loss * (1 - np.exp(-0.1 * epochs))
val_trend = initial_loss * np.exp(-0.13 * epochs) + final_loss * (1 - np.exp(-0.13 * epochs))
test_trend = initial_loss * np.exp(-0.12 * epochs) + final_loss * (1 - np.exp(-0.12 * epochs))
noise_t = 0.2 * np.random.randn(len(epochs))
noise_v = 0.2 * np.random.randn(len(epochs))
noise_s = 0.2 * np.random.randn(len(epochs))

train_loss = trend + noise_t
val_loss = val_trend + noise_v * 0.3
test_loss = test_trend + noise_s * 0.55


train_loss[train_loss < 0.15] = 0.15
train_loss[train_loss > initial_loss + 0.1] = initial_loss + 0.1
val_loss[val_loss < 0.15] = 0.15
val_loss[val_loss > initial_loss + 0.1] = initial_loss + 0.1
test_loss[test_loss < 0.15] = 0.15
test_loss[test_loss > initial_loss + 0.1] = initial_loss + 0.1

plt.figure(figsize=(18, 12))
plt.plot(epochs, train_loss, 'b-', label='Train loss', linewidth=2)
plt.plot(epochs, val_loss, 'r--', label='Val loss', linewidth=2)
plt.plot(epochs, test_loss, 'g-.', label='test loss', linewidth=2)

plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Model loss curve', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper right')
plt.axhline(y=0.2, color='k', linestyle='--', alpha=0.5, linewidth=1)

plt.show()


