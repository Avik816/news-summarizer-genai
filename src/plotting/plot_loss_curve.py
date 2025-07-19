import matplotlib.pyplot as mplot
from ..CONFIG import LOSS_CURVE_PATH
import datetime


def plot_train_val_loss(history):
    # Plotting training vs validation loss
    mplot.plot(history.history['loss'], label='Training Loss')
    mplot.plot(history.history['val_loss'], label='Validation Loss')
    mplot.title('Loss Curve')
    mplot.xlabel('Epochs')
    mplot.ylabel('Loss')
    mplot.legend()
    mplot.grid(True)
    mplot.tight_layout()
    
    # Saving the file
    mplot.savefig(f'{LOSS_CURVE_PATH}/Model learning curve_loss-plot {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.png')
    print(f'Plot is saved at {LOSS_CURVE_PATH} !')