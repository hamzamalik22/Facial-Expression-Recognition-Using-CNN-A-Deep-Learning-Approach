import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import FER2013, get_transforms
from models import VGG
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(epoch, net, trainloader, optimizer, criterion):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0; correct = 0; total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%')

def test(epoch, net, testloader, criterion):
    net.eval()
    # Note: Using TenCrop requires averaging predictions over the 10 crops [cite: 125, 184]
    test_loss = 0; correct = 0; total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            bs, ncrops, c, h, w = inputs.size()
            outputs = net(inputs.view(-1, c, h, w))
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            
            loss = criterion(outputs_avg, targets)
            test_loss += loss.item()
            _, predicted = outputs_avg.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f'Test Accuracy: {100.*correct/total:.3f}%')

# Add main execution block here to initialize data and run loops