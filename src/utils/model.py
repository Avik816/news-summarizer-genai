import torch
from transformers import T5ForConditionalGeneration
import pytorch_lightning as pl
from ..CONFIG import MODEL_NAME, LEARNING_RATE, FACTOR, PATIENCE_ON_RLR, FREQUENCY
from datetime import datetime


class T5FineTuner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.lr = LEARNING_RATE
        self._freeze_layers()

    def _freeze_layers(self):
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 decoder blocks
        for block in self.model.decoder.block[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze lm_head
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)

        # Logging training loss
        self.log('train_loss', outputs.loss, on_step=True, on_epoch=True, prog_bar=True)

        # Logging learning rate manually
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=True)
        
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)

        # Logging val loss
        self.log('val_loss', outputs.loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr
        )

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=FACTOR,
                patience=PATIENCE_ON_RLR,
                verbose=True
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': FREQUENCY
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
