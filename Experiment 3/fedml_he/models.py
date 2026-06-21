"""Model zoo for the overhead-scaling curve.

A registry name -> builder spanning ~0.1M to ~86M parameters:
  SmallCNN  (~0.1M)  tiny CIFAR conv net
  LSTM      (~2M)    small word-level LSTM language model (Penn-Treebank-shaped)
  ResNet-18/34/50    torchvision, CIFAR-adapted stem (3x3 conv, no maxpool)
  ViT-B/16  (~86M)   torchvision vision transformer (BERT-scale parameter count)

The ResNets use a CIFAR stem so the same builder is valid both for parameter
counting (Experiments A/B) and for real CIFAR-10 training (Experiment C).
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class SmallCNN(nn.Module):
    """Compact CIFAR-10 conv net (~0.1M params) -- the small end of the curve."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SmallLSTM(nn.Module):
    """Word-level LSTM language model (~2M params), Penn-Treebank-shaped."""
    def __init__(self, vocab=10000, embed=200, hidden=200, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        e = self.embed(x)
        out, _ = self.lstm(e)
        return self.head(out)


def _cifar_resnet(builder, num_classes=10):
    """torchvision ResNet with a CIFAR stem (3x3 stride-1 conv, no maxpool)."""
    m = builder(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def _vit(num_classes=10):
    """torchvision ViT-B/16 (~86M params); image_size kept at the default 224."""
    return tvm.vit_b_16(num_classes=num_classes)


BUILDERS = {
    "SmallCNN":  SmallCNN,
    "LSTM":      SmallLSTM,
    "ResNet-18": lambda: _cifar_resnet(tvm.resnet18),
    "ResNet-34": lambda: _cifar_resnet(tvm.resnet34),
    "ResNet-50": lambda: _cifar_resnet(tvm.resnet50),
    "ViT-B/16":  _vit,
}


def build_model(name):
    if name not in BUILDERS:
        raise KeyError(f"unknown model {name!r}; have {list(BUILDERS)}")
    return BUILDERS[name]()


def num_params(model):
    return sum(p.numel() for p in model.parameters())
