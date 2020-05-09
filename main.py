import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from dataset import HandDataset
from network import Net

CSV_NAME = 'HandInfo.csv'
DATASET_NAME = 'Hands'

dset = HandDataset(DATASET_NAME, CSV_NAME)
print('######### Dataset class created #########')
print('Number of images: ', len(dset))
print('Sample image shape: ', dset[0]['image'].shape, end='\n\n')

dataloader = torch.utils.data.DataLoader(dset, batch_size=4, shuffle=True, num_workers=4)
net = Net()
print('######### Network created #########')
print('Architecture:\n', net)

### Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):

    running_loss = 0.0
    examples = 0
    for i, data in enumerate(dataloader, 0):
        # Get the inputs
        inputs, labels = data['image'], data['class']

        # Wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.data[0]
        examples += 4
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / examples))

print('Finished Training')