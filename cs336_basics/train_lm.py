import torch
import math
import einx
from jaxtyping import Bool, Float, Int
from collections.abc import Callable, Iterable
from typing import Optional
from torch import Tensor
from .transformer import *

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    max_value = torch.max(inputs, dim=-1, keepdim=True).values
    inputs_normed = inputs - max_value
    exp_value = torch.exp(inputs_normed)
    minus_log_softmax = torch.log(torch.sum(exp_value, dim=-1, keepdim=True)) - einx.get_at('... [vocab_size], ... -> ...', inputs_normed, targets)
    return torch.mean(minus_log_softmax)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data # Get the gradient of loss with respect to p.
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                alpha_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.data -= alpha_t * (m / (torch.sqrt(v) + eps)) # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        delta_it = cosine_cycle_iters - warmup_iters
        delta_learning_rate = max_learning_rate - min_learning_rate
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / delta_it * math.pi)) * delta_learning_rate
    else:
        return min_learning_rate
        

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    params = list(parameters)
    eps = 1e-6
    l2_sum = 0
    for p in params:
        if p.grad is None:
            continue
        l2_sum += torch.sum(p.grad * p.grad).cpu()
    l2_norm = math.sqrt(l2_sum)
    if l2_norm < max_l2_norm:
        return
    for p in params:
        if p.grad is None:
            continue
        p.grad.data *= max_l2_norm / (l2_norm + eps)

if __name__ == '__main__':
    pass
