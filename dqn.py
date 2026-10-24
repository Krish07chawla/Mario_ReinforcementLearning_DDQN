import torch
from torch import nn


class MarioNet(nn.Module):
    """input (4,84,84) -> 3 conv -> 2 dense -> one Q-value per action"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.online = self._build_cnn(c, output_dim)
        self.target = self._build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():      # target is frozen
            p.requires_grad = False

    def forward(self, x, model):
        x = x.float() / 255.0                   # uint8 -> 0..1
        return self.online(x) if model == "online" else self.target(x)

    def _build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, output_dim),
        )
