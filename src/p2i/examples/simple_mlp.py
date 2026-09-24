from torch import nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.layers(x)
