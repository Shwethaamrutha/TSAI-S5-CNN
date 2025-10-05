# Optimized MNIST CNN - Target: 99.4% accuracy with <20k parameters in <20 epochs
# Required: Batch Normalization, Dropout, Global Average Pooling

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# Install required packages
!pip install torchsummary
from torchsummary import summary

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# Model Definitions - All 5 Models

# Model 1: Original OptimizedNet (Baseline)
class OptimizedNet(nn.Module):
    """
    Original optimized architecture
    - Minimal channels for parameter efficiency
    - Strategic dropout placement
    - Global Average Pooling for parameter reduction
    """
    def __init__(self, dropout_rate=0.1):
        super(OptimizedNet, self).__init__()
        
        # Block 1: Initial feature extraction (minimal channels)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Block 2: Feature expansion (still minimal)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # 28x28 -> 28x28
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Transition layer: 1x1 conv for dimensionality reduction
        self.transition1 = nn.Conv2d(16, 8, 1)  # 28x28 -> 28x28
        self.bn_trans1 = nn.BatchNorm2d(8)
        
        # MaxPool after transition
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Block 3: Mid-level features (moderate channels)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)  # 14x14 -> 14x14
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Block 4: Feature expansion (moderate channels)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)  # 14x14 -> 14x14
        self.bn4 = nn.BatchNorm2d(32)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Transition layer: 1x1 conv for dimensionality reduction
        self.transition2 = nn.Conv2d(32, 16, 1)  # 14x14 -> 14x14
        self.bn_trans2 = nn.BatchNorm2d(16)
        
        # MaxPool after transition
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Block 5: High-level features (moderate channels)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)  # 7x7 -> 7x7
        self.bn5 = nn.BatchNorm2d(32)
        self.dropout5 = nn.Dropout2d(dropout_rate)
        
        # Block 6: Final feature extraction (reduce for GAP)
        self.conv6 = nn.Conv2d(32, 16, 3, padding=1)  # 7x7 -> 7x7
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout6 = nn.Dropout2d(dropout_rate)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 7x7 -> 1x1
        
        # Final classifier
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x):
        # Block 1
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        
        # Transition 1
        x = F.relu(self.bn_trans1(self.transition1(x)))
        x = self.pool1(x)
        
        # Block 3
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        
        # Transition 2
        x = F.relu(self.bn_trans2(self.transition2(x)))
        x = self.pool2(x)
        
        # Block 5
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))
        
        # Block 6
        x = self.dropout6(F.relu(self.bn6(self.conv6(x))))
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Final classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Model 2: Improved Architecture (Winner)
class OptimizedNetV2(nn.Module):
    """
    Improved architecture with better channel progression
    - Better initial learning dynamics
    - Controlled capacity for parameter constraint
    - Optimized dropout rates
    """
    def __init__(self, dropout_rate=0.05):
        super(OptimizedNetV2, self).__init__()
        
        # Block 1: Better initial feature extraction (reduced channels)
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(10)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Block 2: Feature expansion with controlled capacity
        self.conv2 = nn.Conv2d(10, 16, 3, padding=1)  # 28x28 -> 28x28
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Transition layer: 1x1 conv for dimensionality reduction
        self.transition1 = nn.Conv2d(16, 10, 1)  # 28x28 -> 28x28
        self.bn_trans1 = nn.BatchNorm2d(10)
        
        # MaxPool after transition
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Block 3: Mid-level features with controlled capacity
        self.conv3 = nn.Conv2d(10, 20, 3, padding=1)  # 14x14 -> 14x14
        self.bn3 = nn.BatchNorm2d(20)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Block 4: Feature expansion
        self.conv4 = nn.Conv2d(20, 28, 3, padding=1)  # 14x14 -> 14x14
        self.bn4 = nn.BatchNorm2d(28)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Transition layer: 1x1 conv for dimensionality reduction
        self.transition2 = nn.Conv2d(28, 16, 1)  # 14x14 -> 14x14
        self.bn_trans2 = nn.BatchNorm2d(16)
        
        # MaxPool after transition
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Block 5: High-level features
        self.conv5 = nn.Conv2d(16, 24, 3, padding=1)  # 7x7 -> 7x7
        self.bn5 = nn.BatchNorm2d(24)
        self.dropout5 = nn.Dropout2d(dropout_rate)
        
        # Block 6: Final feature extraction
        self.conv6 = nn.Conv2d(24, 16, 3, padding=1)  # 7x7 -> 7x7
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout6 = nn.Dropout2d(dropout_rate)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 7x7 -> 1x1
        
        # Final classifier
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x):
        # Block 1
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        
        # Transition 1
        x = F.relu(self.bn_trans1(self.transition1(x)))
        x = self.pool1(x)
        
        # Block 3
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        
        # Transition 2
        x = F.relu(self.bn_trans2(self.transition2(x)))
        x = self.pool2(x)
        
        # Block 5
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))
        
        # Block 6
        x = self.dropout6(F.relu(self.bn6(self.conv6(x))))
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Final classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Model 3: Balanced Architecture with SGD Optimization
class OptimizedNetV3(nn.Module):
    """
    Balanced architecture optimized for SGD training
    - Moderate dropout rates (0.01-0.1)
    - Better learning dynamics
    - Optimized for SGD optimizer
    """
    def __init__(self, dropout_rate=0.02):
        super(OptimizedNetV3, self).__init__()
        
        # Block 1: Initial feature extraction
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(10)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Block 2: Feature expansion
        self.conv2 = nn.Conv2d(10, 16, 3, padding=1)  # 28x28 -> 28x28
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Transition layer: 1x1 conv for dimensionality reduction
        self.transition1 = nn.Conv2d(16, 10, 1)  # 28x28 -> 28x28
        self.bn_trans1 = nn.BatchNorm2d(10)
        
        # MaxPool after transition
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Block 3: Mid-level features
        self.conv3 = nn.Conv2d(10, 20, 3, padding=1)  # 14x14 -> 14x14
        self.bn3 = nn.BatchNorm2d(20)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Block 4: Feature expansion
        self.conv4 = nn.Conv2d(20, 32, 3, padding=1)  # 14x14 -> 14x14
        self.bn4 = nn.BatchNorm2d(32)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Transition layer: 1x1 conv for dimensionality reduction
        self.transition2 = nn.Conv2d(32, 16, 1)  # 14x14 -> 14x14
        self.bn_trans2 = nn.BatchNorm2d(16)
        
        # MaxPool after transition
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Block 5: High-level features
        self.conv5 = nn.Conv2d(16, 24, 3, padding=1)  # 7x7 -> 7x7
        self.bn5 = nn.BatchNorm2d(24)
        self.dropout5 = nn.Dropout2d(dropout_rate)
        
        # Block 6: Final feature extraction
        self.conv6 = nn.Conv2d(24, 16, 3, padding=1)  # 7x7 -> 7x7
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout6 = nn.Dropout2d(dropout_rate)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 7x7 -> 1x1
        
        # Final classifier
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x):
        # Block 1
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        
        # Transition 1
        x = F.relu(self.bn_trans1(self.transition1(x)))
        x = self.pool1(x)
        
        # Block 3
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        
        # Transition 2
        x = F.relu(self.bn_trans2(self.transition2(x)))
        x = self.pool2(x)
        
        # Block 5
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))
        
        # Block 6
        x = self.dropout6(F.relu(self.bn6(self.conv6(x))))
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Final classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Model 4: Model1 Architecture with SGD Optimization
class OptimizedNetV4(nn.Module):
    """
    Model1 architecture optimized for SGD training
    - Same architecture as Model1 (8→16→32→16)
    - Minimal dropout (0.02) for SGD compatibility
    - Optimized for SGD with momentum
    """
    def __init__(self, dropout_rate=0.02):
        super(OptimizedNetV4, self).__init__()
        
        # Block 1: Initial feature extraction (minimal channels)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Block 2: Feature expansion (still minimal)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # 28x28 -> 28x28
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Transition layer: 1x1 conv for dimensionality reduction
        self.transition1 = nn.Conv2d(16, 8, 1)  # 28x28 -> 28x28
        self.bn_trans1 = nn.BatchNorm2d(8)
        
        # MaxPool after transition
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Block 3: Mid-level features (moderate channels)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)  # 14x14 -> 14x14
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Block 4: Feature expansion (moderate channels)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)  # 14x14 -> 14x14
        self.bn4 = nn.BatchNorm2d(32)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Transition layer: 1x1 conv for dimensionality reduction
        self.transition2 = nn.Conv2d(32, 16, 1)  # 14x14 -> 14x14
        self.bn_trans2 = nn.BatchNorm2d(16)
        
        # MaxPool after transition
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Block 5: High-level features (moderate channels)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)  # 7x7 -> 7x7
        self.bn5 = nn.BatchNorm2d(32)
        self.dropout5 = nn.Dropout2d(dropout_rate)
        
        # Block 6: Final feature extraction (reduce for GAP)
        self.conv6 = nn.Conv2d(32, 16, 3, padding=1)  # 7x7 -> 7x7
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout6 = nn.Dropout2d(dropout_rate)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 7x7 -> 1x1
        
        # Final classifier
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x):
        # Block 1
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        
        # Transition 1
        x = F.relu(self.bn_trans1(self.transition1(x)))
        x = self.pool1(x)
        
        # Block 3
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        
        # Transition 2
        x = F.relu(self.bn_trans2(self.transition2(x)))
        x = self.pool2(x)
        
        # Block 5
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))
        
        # Block 6
        x = self.dropout6(F.relu(self.bn6(self.conv6(x))))
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Final classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Model 5: Model2 Architecture with SGD Optimization
class OptimizedNetV5(nn.Module):
    """
    Model2 architecture optimized for SGD training
    - Same architecture as Model2 (10→16→28→16)
    - Minimal dropout (0.02) for SGD compatibility
    - Optimized for SGD with momentum
    """
    def __init__(self, dropout_rate=0.02):
        super(OptimizedNetV5, self).__init__()
        
        # Block 1: Better initial feature extraction (reduced channels)
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(10)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Block 2: Feature expansion with controlled capacity
        self.conv2 = nn.Conv2d(10, 16, 3, padding=1)  # 28x28 -> 28x28
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Transition layer: 1x1 conv for dimensionality reduction
        self.transition1 = nn.Conv2d(16, 10, 1)  # 28x28 -> 28x28
        self.bn_trans1 = nn.BatchNorm2d(10)
        
        # MaxPool after transition
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Block 3: Mid-level features with controlled capacity
        self.conv3 = nn.Conv2d(10, 20, 3, padding=1)  # 14x14 -> 14x14
        self.bn3 = nn.BatchNorm2d(20)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        # Block 4: Feature expansion
        self.conv4 = nn.Conv2d(20, 28, 3, padding=1)  # 14x14 -> 14x14
        self.bn4 = nn.BatchNorm2d(28)
        self.dropout4 = nn.Dropout2d(dropout_rate)
        
        # Transition layer: 1x1 conv for dimensionality reduction
        self.transition2 = nn.Conv2d(28, 16, 1)  # 14x14 -> 14x14
        self.bn_trans2 = nn.BatchNorm2d(16)
        
        # MaxPool after transition
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Block 5: High-level features
        self.conv5 = nn.Conv2d(16, 24, 3, padding=1)  # 7x7 -> 7x7
        self.bn5 = nn.BatchNorm2d(24)
        self.dropout5 = nn.Dropout2d(dropout_rate)
        
        # Block 6: Final feature extraction
        self.conv6 = nn.Conv2d(24, 16, 3, padding=1)  # 7x7 -> 7x7
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout6 = nn.Dropout2d(dropout_rate)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 7x7 -> 1x1
        
        # Final classifier
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x):
        # Block 1
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        
        # Transition 1
        x = F.relu(self.bn_trans1(self.transition1(x)))
        x = self.pool1(x)
        
        # Block 3
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        
        # Transition 2
        x = F.relu(self.bn_trans2(self.transition2(x)))
        x = self.pool2(x)
        
        # Block 5
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))
        
        # Block 6
        x = self.dropout6(F.relu(self.bn6(self.conv6(x))))
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Final classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
