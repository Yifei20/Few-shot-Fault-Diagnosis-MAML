import torch
import learn2learn as l2l
from torchvision import datasets, transforms
from datasets.cwru import CWRU, CWRU_RAW
from torch.utils.data import DataLoader
"""
Pre-train a CNN model on one of the working conditions of the CWRU dataset, 
and then fine-tune the model on the other working conditions of the CWRU dataset.
"""
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import random
from models import CNN1D

data_dir = './data'
train_domain = 0
valid_domain = 2
test_domain = 3
num_epchs = 10
seed = 42
cuda = True

# Set the Random Seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set training device, using GPU if available
if cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device_count = torch.cuda.device_count()
    device = torch.device('cuda')
    print('Training MAML with {} GPU(s).'.format(device_count))
else:
    device = torch.device('cpu')
    print('Training MAML with CPU.')

train_dataset = CWRU_RAW(train_domain, data_dir, fft=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

valid_dataset = CWRU_RAW(valid_domain, data_dir, fft=False)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

test_dataset = CWRU_RAW(test_domain, data_dir, fft=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# model = l2l.vision.models.CNN4(output_size=10)
model = CNN1D()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

valid_loss_min = np.Inf
train_epochs_loss = []
valid_epochs_loss = []
train_epochs_acc = []
valid_epochs_acc = []
print("Pre-training the model...")
for epoch in range(num_epchs):
    print(f"Epoch {epoch+1}/{num_epchs}")
    train_loss_sum = 0.0
    valid_loss_sum = 0.0
    train_acc_num = 0.0
    valid_acc_num = 0.0

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * images.size(0)
        train_acc_num += sum(torch.max(logits, dim=1)[1] == labels).cpu()
    train_acc = 100 * train_acc_num/len(train_loader.dataset)
    train_loss = train_loss_sum/len(train_loader.dataset)
    train_epochs_loss.append(train_loss)
    train_epochs_acc.append(train_acc)
    print('Train loss: {:.3f}, Train acc: {:.1f}%'.format(train_loss, train_acc))

    model.eval()
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
            loss = criterion(logits, labels)

        valid_loss_sum += loss.item() * images.size(0)
        valid_acc_num += sum(torch.max(logits, dim=1)[1] == labels).cpu()

    valid_acc= 100 * valid_acc_num/len(valid_loader.dataset)
    valid_loss = valid_loss_sum/len(valid_loader.dataset)
    valid_epochs_loss.append(valid_loss)
    valid_epochs_acc.append(valid_acc)
    print('Valid loss: {:.3f}, Valid acc: {:.1f}%'.format(valid_loss, valid_acc))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.3f} --> {:.3f}).'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), './models/cnn_pretrain.pt')
        valid_loss_min = valid_loss

model.load_state_dict(torch.load('./models/cnn_pretrain.pt'))
model.eval()
test_loss_sum = 0.0
test_acc_num = 0.0
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels)
    test_loss_sum += loss.item() * images.size(0)
    test_acc_num += sum(torch.max(logits, dim=1)[1] == labels).cpu()
test_acc = 100 * test_acc_num/len(test_loader.dataset)
test_loss = test_loss_sum/len(test_loader.dataset)
print('Test loss: {:.3f}, Test acc: {:.1f}%'.format(test_loss, test_acc))

# Fine-tune the model using different working conditions
print("Fine-tuning the model...")

