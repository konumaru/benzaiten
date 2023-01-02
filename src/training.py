import os
from collections import namedtuple

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from torch.nn.utils import clip_grad_norm_

from dataset import get_dataloader
from factory import get_loss, get_lr_scheduler, get_optimizer
from model import get_model


def setup_modules(cfg: DictConfig, device: torch.device):
    """Instantiate modules for training."""
    dataloader = get_dataloader(cfg)
    model = get_model(cfg, device)
    loss_func = get_loss(cfg, model)
    optimizer = get_optimizer(cfg, model)
    lr_scheduler = None
    if cfg.training.use_scheduler:
        lr_scheduler = get_lr_scheduler(cfg, optimizer)
    TrainingModules = namedtuple(
        "TrainingModules",
        ["dataloader", "model", "loss_func", "optimizer", "lr_scheduler"],
    )
    modules = TrainingModules(
        dataloader=dataloader,
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    return modules


def training_step(batch, loss_func, device: torch.device):
    """Perform a training step."""
    source, target = batch
    batch = {"source": source.to(device).float(), "target": target.to(device).long()}
    loss = loss_func(batch)
    return loss


def training_loop(cfg: DictConfig, modules, device: torch.device):
    """Perform training loop."""
    dataloader, model, loss_func, optimizer, lr_scheduler = modules
    model.train()  # turn on train mode
    n_epoch = cfg.training.n_epoch
    for epoch in range(1, n_epoch + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            loss = training_step(batch, loss_func, device)
            epoch_loss += loss.item()
            loss.backward()
            if cfg.training.use_grad_clip:
                clip_grad_norm_(model.parameters(), cfg.training.grad_max_norm)
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch: >4}/{n_epoch}: loss = {epoch_loss:.6f}")


def save_checkpoint(cfg: DictConfig, modules):
    """Save checkpoint."""
    model = modules.model
    model_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.model_dir)
    model_file = os.path.join(model_dir, cfg.training.model_file)
    torch.save(model.state_dict(), model_file)


def main(cfg: DictConfig):
    """Perform model training."""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate modules for training
    modules = setup_modules(cfg, device)

    # perform training loop
    training_loop(cfg, modules, device)

    # save checkpoint
    save_checkpoint(cfg, modules)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
