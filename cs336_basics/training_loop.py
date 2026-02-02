from dataclasses import dataclass
import glob
import torch
import numpy as np
import numpy.typing as npt
import os
import typing
import re
import pickle
from cs336_basics import *

def data_loading(
    x: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    total_length = x.shape[0]
    indices = np.random.randint(total_length - context_length, size=batch_size)
    input = torch.tensor(np.stack([x[i:i+context_length] for i in indices]), dtype=torch.int32, device=device)
    output = torch.tensor(np.stack([x[i+1:i+1+context_length] for i in indices]), dtype=torch.int32, device=device)
    return (input, output)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out_path: str | os.PathLike
):
    tmp_path = str(out_path) + ".tmp"
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }, tmp_path)
    os.replace(tmp_path, out_path)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    data = torch.load(src)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(data['model'])
    if optimizer is not None:
        optimizer.load_state_dict(data['optimizer'])
    return data['iteration']


@dataclass
class TransformerParams:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    theta: float

@dataclass
class AdamWParams:
    lr: float
    weight_decay: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

torch.set_float32_matmul_precision('high')
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from livelossplot import PlotLosses

def train_model(train_data_path: str | os.PathLike,
                val_data_path: str | os.PathLike,
                params: TransformerParams,
                adamw_params: AdamWParams,
                batch_size: int,
                num_iterations: int,
                log_interval: int,
                checkpoint_interval: int,
                compile: bool,
                checkpoint_root: str | os.PathLike | None = None,
                dtype: torch.dtype | None = None,
                device: torch.device | None = None):
    if not os.path.exists(checkpoint_root):
        os.makedirs(checkpoint_root)
    with open(os.path.join(checkpoint_root, f'params.pkl'), 'wb') as f:
        pickle.dump((params, adamw_params, dtype, device), f)

    plotlosses = PlotLosses()
    x = np.memmap(train_data_path, dtype=np.uint16, mode='r')
    x_val = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    model = TransformerLM(
        vocab_size=params.vocab_size,
        context_length=params.context_length,
        num_layers=params.num_layers,
        d_model=params.d_model,
        num_heads=params.num_heads,
        d_ff=params.d_ff,
        theta=params.theta,
        dtype=dtype,
        device=device
    )
    optimizer = AdamW(
        model.parameters(),
        lr=adamw_params.lr,
        weight_decay=adamw_params.weight_decay,
        betas=adamw_params.betas,
        eps=adamw_params.eps
    )
    start_iteration = 1
    if checkpoint_root is not None and os.path.isdir(checkpoint_root):
        ckpt_files = list(glob.glob(os.path.join(checkpoint_root, "checkpoint_iter_*.pt")))
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
            start_iteration = load_checkpoint(latest_ckpt, model, optimizer)
            print(f"Resumed from iteration {start_iteration}.")
    if compile:
        model = torch.compile(model)
    model.train()
    for iteration in range(start_iteration, num_iterations + 1):
        input, output = data_loading(x, batch_size, params.context_length, device)
        optimizer.zero_grad()
        logits = model(input)
        loss = cross_entropy(
            einx.rearrange('b s d -> (b s) d', logits),
            output.flatten()
        )
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()
        if (iteration-1) % log_interval == 0:
            val_loss_accum = 0
            val_iters = 5
            with torch.no_grad():
                for _ in range(val_iters):
                    x_val_input, x_val_output = data_loading(x_val, batch_size, params.context_length, device)
                    val_logits = model(x_val_input)
                    val_loss_accum +=  cross_entropy(
                        einx.rearrange('b s d -> (b s) d', val_logits),
                        x_val_output.flatten()
                    ).item()
            val_loss = val_loss_accum / val_iters
            plotlosses.update({
                'loss': loss.cpu().item(),
                'val_loss': val_loss
            })
            plotlosses.send()
            print(f"[{iteration}/{num_iterations}] Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            del val_logits, val_loss
        print(f"[{iteration}/{num_iterations}]")
        if iteration % checkpoint_interval == 0 or iteration == num_iterations:
            save_checkpoint(model, optimizer, iteration, os.path.join(checkpoint_root, f'checkpoint_iter_{iteration}.pt'))
        
def load_model_from_checkpoint(params_path: str | os.PathLike, checkpoint_path: str | os.PathLike) -> TransformerLM:
    with open(params_path, 'rb') as f:
        params, _, dtype, device = pickle.load(f)
    model = TransformerLM(
        vocab_size=params.vocab_size,
        context_length=params.context_length,
        num_layers=params.num_layers,
        d_model=params.d_model,
        num_heads=params.num_heads,
        d_ff=params.d_ff,
        theta=params.theta,
        dtype=dtype,
        device=device
    )
    load_checkpoint(checkpoint_path, model)
    return model

