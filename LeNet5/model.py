"""Implementation of LeNet5 model, originally propoposed by Yann LeCun and others in 1998.
Check out the paper:http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""
"""Network architecture:
Input: 32*32 image input layer
Layer1: 5x5 CNN with 6 filters
Layer1 Output: 32-5+1=28 28x28x6 (5*5+1)*6=156 parameters
Layer2: Pooling layer: no learnable parameters
Layer2 Output: 14x14x6
Layer3: 5x5 CNN with with 16 filters
Layer3 Output: 14-5+1=10 10x10x16
Layer4: Pooling layer: no learnable parameters
Layer4 Output: 5x5x16 output feature map
Layer5: 5x5 CNN with 120 filters
Layer5 Output: 5-5+1=1 1x1x120 
Layer6: FC layer with 84 neurons.
Layer7: FC layer with 10 output neurons
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.functional as F
class LeNet5Modern(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,stride=1,padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.fc=nn.Linear(400,120) # this 400 came from 5*5*16
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(120,84)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(84,num_classes)
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=out.reshape(out.size(0),-1)
        out=self.fc(out)
        out=self.relu(out)
        out=self.fc1(out)
        out=self.relu1(out)
        out=self.fc2(out)
        return out
class LeNet5Original(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,stride=1,padding=0),
            nn.BatchNorm2d(6),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2,stride=2))
        self.fc=nn.Linear(400,120) # this 400 came from 5*5*16
        self.tanh=nn.Tanh()
        self.fc1=nn.Linear(120,84)
        self.tanh1=nn.Tanh()
        self.fc2=nn.Linear(84,num_classes)
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=out.reshape(out.size(0),-1)
        out=self.fc(out)
        out=self.tanh(out)
        out=self.fc1(out)
        out=self.tanh1(out)
        out=self.fc2(out)
        return out