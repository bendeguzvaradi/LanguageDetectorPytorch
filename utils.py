import torch
from pathlib import Path


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state,
               Path(__file__).parent.absolute() / 'checkpoints' / filename)


def load_checkpoint(checkpoint, model, optimizer):
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
