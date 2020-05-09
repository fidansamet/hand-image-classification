import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from dataset import HandDataset
from network import device, Net
import torch.nn.functional as F



CSV_NAME = 'HandInfo.csv'
DATASET_NAME = 'Hands'
VALIDATION = .2
TEST = .2
SHUFFLE = True
BATCH_SIZE = 16
RANDOM_SEED = 42
EPOCH = 2

# Build dataset
dataset = HandDataset(DATASET_NAME, CSV_NAME)
print('######### Dataset class created #########')
print('Number of images: ', len(dataset))
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)


# TRAIN-VALIDATION-TEST SUBSETS
dataset_size = len(dataset)
indices = list(range(dataset_size))
validation_split = int(np.floor(VALIDATION * dataset_size))
test_split = int(np.floor(TEST * dataset_size))

if SHUFFLE:
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)

train_indices = indices[(validation_split + test_split):]
val_indices = indices[validation_split:(validation_split + test_split)]
test_indices = indices[:validation_split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)


# CREATE NETWORK
net = Net()
net.cuda()
print('######### Network created #########')
print('Architecture:\n', net)


# Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(EPOCH):

    running_loss = 0.0
    examples = 0

    for i, data in enumerate(train_loader, 0):
        # Get the inputs
        inputs, labels = data['image'], data['class']

        # Wrap them in Variable
        # inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device=device, dtype=torch.float), labels.to(device=device, dtype=torch.long)    # TODO

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #loss = criterion(outputs, labels)
        loss = F.nll_loss(outputs, labels)

        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.data
        examples += BATCH_SIZE
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / examples))

print('Finished Training')