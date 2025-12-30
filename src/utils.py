import sys
import time
import matplotlib.pyplot as plt

def progress_bar(current, total, msg=None):
    """The custom progress bar used in your notebook's training loops[cite: 186]."""
    bar_len = 30
    filled_len = int(round(bar_len * current / float(total)))
    percents = round(100.0 * current / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(f'[{bar}] {percents}% ...{msg}\r')
    sys.stdout.flush()

def plot_history(history, title="Model Accuracy"):
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()