from ..CONFIG import TRAIN_DIR, VAL_DIR, BATCH_SIZE, EPOCHS
from ..utils.model import T5FineTuner
from .data_loaders import StreamingNewsSummaryDataset, collate_fn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
import torch


def train_T5Small_model():
    # Downloading and setting up the model for fine-tuning
    model = T5FineTuner()

    train_dataset = StreamingNewsSummaryDataset(TRAIN_DIR)
    val_dataset = StreamingNewsSummaryDataset(VAL_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Model Callbacks
    checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best-t5')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Settting up the Trainer
    t5_small_trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint, early_stop, lr_monitor],
        accelerator='auto',
        log_every_n_steps=10,
        precision=16 if torch.cuda.is_available() else 32,
    )

    t5_small_trainer.fit(model, train_loader, val_loader)