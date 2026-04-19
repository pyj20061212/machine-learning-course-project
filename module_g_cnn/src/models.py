import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dims=(256, 128), num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    """
    A compact CNN for MNIST:
    1x28x28
      -> Conv(1,8,3,pad=1) -> ReLU -> MaxPool(2)
      -> Conv(8,16,3,pad=1) -> ReLU -> MaxPool(2)
      -> Flatten
      -> Linear(16*7*7,64) -> ReLU
      -> Linear(64,10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward_features(self, x):
        feat1 = F.relu(self.conv1(x))
        pooled1 = self.pool1(feat1)

        feat2 = F.relu(self.conv2(pooled1))
        pooled2 = self.pool2(feat2)

        return {
            "conv1": feat1,
            "pool1": pooled1,
            "conv2": feat2,
            "pool2": pooled2,
        }

    def forward(self, x):
        feats = self.forward_features(x)
        x = feats["pool2"]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class AblationCNN(nn.Module):
    """
    Configurable CNN for ablation study.

    Parameters
    ----------
    channels : tuple[int, int]
        Output channels of conv1 and conv2.
    use_pool : bool
        Whether to use max pooling after each conv block.
    hidden_dim : int
        Hidden dimension of classifier.
    num_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        channels=(8, 16),
        use_pool: bool = True,
        hidden_dim: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        c1, c2 = channels
        self.use_pool = use_pool

        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if use_pool:
            feat_dim = c2 * 7 * 7
        else:
            feat_dim = c2 * 28 * 28

        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward_features(self, x):
        feat1 = F.relu(self.conv1(x))
        x1 = self.pool(feat1) if self.use_pool else feat1

        feat2 = F.relu(self.conv2(x1))
        x2 = self.pool(feat2) if self.use_pool else feat2

        return {
            "conv1": feat1,
            "block1_out": x1,
            "conv2": feat2,
            "block2_out": x2,
        }

    def forward(self, x):
        feats = self.forward_features(x)
        x = feats["block2_out"]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)