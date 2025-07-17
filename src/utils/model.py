import torch
from transformers import T5ForConditionalGeneration
import pytorch_lightning as pl
from ..CONFIG import MODEL_NAME, LEARNING_RATE


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
        self.log("train_loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)