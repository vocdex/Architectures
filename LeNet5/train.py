import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from LeNet5.model import LeNet5Modern

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Loading the dataset
batch_size=64
num_classes=10
#Loading the dataset and preprocessing
train_dataset = torchvision.datasets.MNIST(root = './data',
                                           train = True,
                                           transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                           download = True)
test_dataset = torchvision.datasets.MNIST( root = './data',
                                          train = False,
                                          transform = transforms.Compose([
                                                  transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                          download=True)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

# Setting hyperparameters

learning_rate=0.001
num_epochs=16
# Model Selection
model=LeNet5Modern(num_classes).to(device)
# change to LeNet5Original if you want to use the original model
cost=nn.CrossEntropyLoss()
# cost=nn.MSELoss() in original model
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
#optimizer= SGD(model.parameters(), lr=1e-1) in original model
total_step=len(train_loader)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)
        # Forward pass
        outputs=model(images)
        loss=cost(outputs,labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%400==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))