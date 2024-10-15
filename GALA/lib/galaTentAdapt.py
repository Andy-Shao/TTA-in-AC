import torch 
import torch.nn as nn
import torch.jit
import torch.optim as optim

from lib.tentAdapt import adapted_loss_fn

@torch.enable_grad()
def adapted_forward(
    x: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, 
    lr:float, threshold:float, selected:bool
) -> torch.Tensor:
    outputs = model(x)
    optimizer.zero_grad()
    loss = adapted_loss_fn(outputs).mean(0)
    loss.backward()

    if selected:
        with torch.no_grad():
            for component in model.modules():
                if isinstance(component, nn.BatchNorm2d):
                    if consine_similarity(module=component, lr=lr) > threshold:
                        update_param(component, lr=lr)
    else:
        optimizer.step()

    return outputs

def update_param(module: nn.Module, lr:float) -> None:
    for param in module.parameters():
        if param.grad is not None:
            if param.requires_grad:
                param -= param.grad * lr
            else:
                param.grad.zero_()

def consine_similarity(module: nn.Module, lr:float):
    mus = []
    graded_mus = []
    with torch.no_grad():
        for param in module.parameters():
            if param.grad is not None:
                if param.requires_grad:
                    mu = torch.empty_like(param).copy_(param)
                    graded_mu = mu - (param.grad * lr)
                    mus.append(mu)
                    graded_mus.append(graded_mu)
    mus = torch.cat(mus, dim=0)
    graded_mus = torch.cat(graded_mus, dim=0)
    return mus @ graded_mus / (torch.norm(mus, p=2) * torch.norm(graded_mus, p=2))