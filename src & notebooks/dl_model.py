import torch.nn as nn
import torch.nn.functional as F

class SVHN_CNN(nn.Module):
    def __init__(self):
        super(SVHN_CNN, self).__init__()
        # Convolutional Block 1: Learn edges/simple textures
        # Input: (3, 32, 32) -> Output: (32, 32, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Normalization for faster training
        
        # Convolutional Block 2: Learn shapes/curves
        # Input: (32, 16, 16) -> Output: (64, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pooling reduces size by half (32->16->8)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25) # Regularization

        # Fully Connected Layers (Classifier)
        # Input flattened: 64 channels * 8 * 8 size = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10) # 10 Classes

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten
        x = x.view(-1, 64 * 8 * 8)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x