# CNN-basic with MNIST DATASET
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# Hyperparameters
random_seed=1
learning_rate=0.05
num_epochs=10
batch_size=128

# Architecture
num_classes=10
####################
#### MNIST DATASET
####################
# transforms.ToTensor() scales input images to 0-1 range
train_dataset=datasets.MNIST(root='data',train=True,
                             transform=transforms.ToTensor(),download=True)
test_dataset=datasets.MNIST(root='data',train=False,transform=transforms.ToTensor())

train_loader=DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True)
test_loader=DataLoader(dataset=test_dataset,
                       batch_size=batch_size,
                       shuffle=False)
# Checking the dataset
for images,labels in train_loader:
    print("Image batch dimensions:",images.shape)
    print("Image label dimensions:",labels.shape)
    break


###################
### MODEL
###################
# we are just manually creating dimensions for the next layers by specifiying the output
# channel, kernel, and stride size. Using these values, we are calculating
# a corresponding padding size : p=(s(o-1)-w+k)/2
class ConvNet(torch.nn.Module):

    def __init__(self,num_classes):
        super(ConvNet, self).__init__()
        # calculate same padding:
        # (w-k+2*p)/s +1=0
        # => p=(s(o-1)-w+k)/2

        # 28*28*1 => 28*28*8
        self.conv_1=torch.nn.Conv2d(in_channels=1,
                                    out_channels=8,
                                    kernel_size=(3,3),
                                    stride=(1,1),
                                    padding=1) #(1(28-1)-28+3)/2
        # 28*28*8 => 14*14*8
        self.pool_1=torch.nn.MaxPool2d(kernel_size=(2,2),
                                       stride=(2,2),
                                       padding=0) #(2*(14-1)-28+2)=0
        #14*14*8 => 14*14*16
        self.conv_2=torch.nn.Conv2d(in_channels=8,out_channels=16,
                                    kernel_size=(3,3),
                                    stride=(1,1),
                                    padding=1)# (1(14-1)-14+3)/2
        # 14*14*16 => 7*7*16
        self.pool_2=torch.nn.MaxPool2d(kernel_size=(2,2),
                                       stride=(2,2),
                                       padding=0)
        self.linear_1=torch.nn.Linear(7*7*16,num_classes)
    def forward(self,x):
        out=self.conv_1(x)
        out=F.relu(out)
        out=self.pool_1(out)

        out=self.conv_2(out)
        out=F.relu(out)
        out=self.pool_2(out)
        logits=self.linear_1(out.view(-1,7*7*16))
        probas=F.softmax(logits,dim=1)
        return logits,probas
torch.manual_seed(random_seed)
model=ConvNet(num_classes=num_classes)
model=model.to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

# TRAINING
def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 1000
start_time=time.time()

for epoch in range(num_epochs):
    model=model.train()
    for batch_idx,(features,targets) in enumerate(train_loader):
        features=features.to(device)
        targets=targets.to(device)
        ### Forward and BACKPROP
        logits,probas=model(features)
        cost=F.cross_entropy(logits,targets)
        optimizer.zero_grad()
        cost.backward()
        # Update model parameters
        optimizer.step()
        ### Logging
        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1, num_epochs, batch_idx,
                     len(train_loader), cost))
    model=model.eval()
    print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
        epoch + 1, num_epochs,
        compute_accuracy(model, train_loader)))
    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))