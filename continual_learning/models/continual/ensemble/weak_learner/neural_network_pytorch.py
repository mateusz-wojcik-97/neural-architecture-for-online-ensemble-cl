from typing import List, Any, Optional, Dict

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from sklearn.neural_network import MLPClassifier
from torch import Tensor, from_numpy
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torchmetrics import Accuracy

from continual_learning.models.continual.ensemble.weak_learner.base import WeakLearner
from continual_learning.optimizers.naive_optimizer import NaiveOptimizer


class NeuralNetworkPyTorch(WeakLearner):
    def __init__(self, classes: List[int], **kwargs):
        self.classes = classes
        self.model = MNISTModel(classes=len(classes), **kwargs)
        self.trainer = Trainer(
            max_epochs=1,
            progress_bar_refresh_rate=0,
            # enable_progress_bar=True,
            enable_checkpointing=False,
            checkpoint_callback=False,
            logger=False,
            weights_summary=None,
            precision=16,
            # profiler="simple",
        )

    def fit(self, data_loader: DataLoader, clf_meta: Optional[Dict[str, Any]] = None):
        self.model.configure_call_meta(clf_meta)
        self.trainer.fit(self.model, data_loader)
        self.trainer.fit_loop.max_epochs += 1

    def predict(self, data_loader: DataLoader, clf_meta: Optional[Dict[str, Any]] = None) -> Tensor:
        self.model.configure_call_meta(clf_meta)
        return self.trainer.predict(self.model, data_loader)


class MNISTModel(LightningModule):
    def __init__(self, learning_rate: float, classes: int, input_size: int):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = 0.0001  # TODO move into argument
        self.tanh_factor = 250  # TODO move into argument
        self.classes = classes
        self.input_size = input_size
        self.linear = torch.nn.Linear(self.input_size, classes)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self._call_meta = {}

    def configure_call_meta(self, meta: Dict[str, Any]) -> None:
        self._call_meta = meta

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return NaiveOptimizer(
            self.parameters(), learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )

    def activate_softmax(self, raw_output: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.linear(raw_output))  # TODO disable grad

    def activate_tanh(self, raw_output: torch.Tensor) -> torch.Tensor:
        return self.tanh(self.linear(raw_output) / self.tanh_factor) * self.tanh_factor

    def forward(self, x):
        raw_output = x.view(x.size(0), -1)
        vanilla_output = self.activate_softmax(raw_output)
        tanh_output = self.activate_tanh(raw_output)
        return vanilla_output.squeeze(), tanh_output.squeeze()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        vanilla_output, tanh_output = self(batch)
        return torch.argmax(tanh_output, dim=0)

    def loss(self, vanilla_output: torch.Tensor, tanh_output: torch.Tensor, label: torch.Tensor):
        # Label to one hot
        label_one_hot = torch.zeros(self.classes)
        label_one_hot[label.item()] = 1
        # Classifier loss
        vanilla_loss = -torch.dot(vanilla_output, label_one_hot)
        tanh_loss = -torch.dot(tanh_output, label_one_hot)
        # Memory loss TODO check
        cosine_similarities = torch.Tensor(self._call_meta['cos_similarity'])
        cosine_similarity = cosine_similarities[self._call_meta['index']]
        mean_value = tanh_output * cosine_similarity / torch.sum(cosine_similarities, dim=0)
        mean_value = mean_value.squeeze()
        memory_loss = -torch.dot(mean_value, label_one_hot)
        # Total loss
        loss = vanilla_loss + tanh_loss + memory_loss
        return loss

    def training_step(self, batch, batch_nb):
        x, y = batch
        vanilla_output, tanh_output = self(x)
        # _y = torch.zeros(self.classes)
        # _y[y.item()] = 1
        # loss = -torch.dot(tanh_output.squeeze(), _y)

        loss = self.loss(vanilla_output, tanh_output, y)

        # preds = torch.argmax(logits, dim=1)
        # self.train_accuracy(preds, y)
        # self.log("train_loss", loss, prog_bar=True)
        # self.log("train_acc", self.train_accuracy, prog_bar=True)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.cross_entropy(self(x), y)
    #     preds = torch.argmax(logits, dim=1)
    #     self.test_accuracy(preds, y)
    #
    #     self.log("val_loss", loss, prog_bar=True)
    #     self.log("val_acc", self.test_accuracy, prog_bar=True)
    #     return loss
    #
    # def test_step(self, batch, batch_idx):
    #     return self.validation_step(batch, batch_idx)


if __name__ == '__main__':
    pass
