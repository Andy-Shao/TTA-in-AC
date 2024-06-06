from copy import deepcopy

import torch 
import torch.nn as nn
import torch.jit

class TentAdapt(nn.Module):
    """Tent test-time adaptation"""
    def __init__(self, model: nn.Module, optimizer, steps=1, episodic=False) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # TODO

    def reset(self):
        pass

def model_formate(model: nn.Module) -> nn.Module:
    """Formate the model for tent test-time adaptation"""
    model.train()
    model.requires_grad_(False)
    for component in model.modules():
        if isinstance(component, nn.BatchNorm2d):
            component.requires_grad_(True)
            # force use of batch stats in train and eval modes
            component.track_running_stats = False
            component.running_mean = None
            component.running_var = None
    return model