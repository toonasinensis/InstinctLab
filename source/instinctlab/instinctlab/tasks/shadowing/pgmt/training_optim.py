"""Muon on selected linear matrices, retaining Adam on the remaining parameters.

Uses PartyMimic's parameter grouping and Adam-RMS learning-rate convention:
https://github.com/Roboparty/PartyMimic/blob/536be97/src/partymimic/optim.py
"""

import torch
from torch import nn


class MuonAdam:
    def __init__(self, module, matrices, lr, betas, weight_decay=0.0):
        selected = {id(child.weight) for layer in matrices for child in layer.modules() if isinstance(child, nn.Linear)}
        self.muon = torch.optim.Muon(
            [p for p in module.parameters() if id(p) in selected],
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
        adam = torch.optim.AdamW if weight_decay else torch.optim.Adam
        self.adam = adam(
            [p for p in module.parameters() if id(p) not in selected], lr=lr, betas=betas, weight_decay=weight_decay
        )

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adam.step()

    def state_dict(self):
        return dict(muon=self.muon.state_dict(), adam=self.adam.state_dict())

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])
