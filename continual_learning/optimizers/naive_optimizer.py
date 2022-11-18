import torch
from torch.optim.optimizer import Optimizer

import numpy as np


class NaiveOptimizer(Optimizer):
    def __init__(self, params, learning_rate: float, weight_decay: float, device: torch.device):
        defaults = dict(learning_rate=learning_rate, weight_decay=weight_decay)
        super(NaiveOptimizer, self).__init__(params, defaults)
        self.updates_count = 0
        self.grad_none = 0
        self.device = device

    def __setstate__(self, state):
        super(NaiveOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    self.grad_none += 1
                    continue

                self.updates_count += 1

                update = -torch.sign(p.grad)
                update += group['weight_decay'] * p
                update *= group['learning_rate']

                if_yes = torch.tensor(0.0).to(self.device)
                if_no = torch.tensor(1.0).to(self.device)
                mask = torch.where(p.grad == 0, if_yes, if_no)
                update *= mask

                p.data.add_(update)

        return loss
