from ..CONFIG import TRAIN_DIR, VAL_DIR, BATCH_SIZE, CHECKPOINT_PATH, EPOCHS, MODEL_PATH, CSV_LOGGER_PATH
from ..utils.tokenizer import get_tokenizer
from ..utils.model import T5FineTuner
from .dataset_streamer import StreamingNewsSummaryDataset
from ..utils.custom_batch_collation import collate_fn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import torch
from datetime import datetime
from ..plotting.plot_loss_curve import plot_train_val_loss
from transformers import logging as hf_logging


def train_T5Small_model():
    hf_logging.set_verbosity_warning()

    # Fetching the tokenizer
    tokenizer = get_tokenizer()

    # Downloading and setting up the model for fine-tuning
    model = T5FineTuner()
    
    # Print the model summary
    # print(model)

    # Creating the datasets for the model
    # This snippet will load the datasets dynamically and not directly in the memory, thus saving space
    train_dataset = StreamingNewsSummaryDataset(TRAIN_DIR, tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    val_dataset = StreamingNewsSummaryDataset(VAL_DIR, tokenizer)   
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    # Model Callbacks
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath=CHECKPOINT_PATH,
        filename="checkpoint_epoch-{epoch:02d}_val_loss-{val_loss:.6f}_lr-{learning_rate:.6f}"
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    csv_logger = CSVLogger('logs', name=CSV_LOGGER_PATH, version='v1')
    progress_bar = RichProgressBar(refresh_rate=1)

    # Settting up the Trainer
    t5_small_trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint, early_stop, lr_monitor, progress_bar],
        accelerator='auto',
        log_every_n_steps=0,
        logger=csv_logger,
        precision=16 if torch.cuda.is_available() else 32,
    )

    # Training the model
    history = t5_small_trainer.fit(model, train_loader, val_loader)

    # Saving the model and tokenizer
    model.model.save_pretrained(MODEL_PATH)

    # Plotting training vs validation loss curve
    plot_train_val_loss(history)
    