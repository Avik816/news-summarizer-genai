from ..CONFIG import TRAIN_DIR, VAL_DIR, BATCH_SIZE, CHECKPOINT_PATH, EPOCHS, MODEL_PATH, CSV_LOGGER_PATH
from ..utils.model import T5FineTuner
from .dataset_streamer import StreamingNewsSummaryDataset
from ..utils.custom_batch_collation import collate_fn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import torch
from datetime import datetime
from ..plotting.plot_loss_curve import plot_train_val_loss


def train_T5Small_model():
    # Downloading and setting up the model for fine-tuning
    model = T5FineTuner()
    
    # Print the model summary
    print(model.summary())

    # Creating the datasets for the model
    # This snippet will load the datasets dynamically and not directly in the memory, thus saving space
    train_dataset = StreamingNewsSummaryDataset(TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    val_dataset = StreamingNewsSummaryDataset(VAL_DIR)   
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Model Callbacks
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        filename=f'{CHECKPOINT_PATH}/checkpoint_{datetime.now().strftime('%Y%m%d-%H%M%S')}_epoch-{{epoch:02d}}_val_loss-{{val_loss:.6f}}'
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    csv_logger = CSVLogger('logs', name=CSV_LOGGER_PATH)

    # Settting up the Trainer
    t5_small_trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint, early_stop, lr_monitor],
        accelerator='auto',
        log_every_n_steps=1,
        logger=csv_logger,
        precision=16 if torch.cuda.is_available() else 32,
    )

    # Training the model
    history = t5_small_trainer.fit(model, train_loader, val_loader)

    # Saving the model and tokenizer
    model.model.save_pretrained(MODEL_PATH)

    # Plotting training vs validation loss curve
    plot_train_val_loss(history)
    