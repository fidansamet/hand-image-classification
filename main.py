import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
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
BATCH_SIZE = 32
RANDOM_SEED = 42
EPOCH = 20
LR = 0.1


def main():
    # BUILD DATASET
    dataset = HandDataset(DATASET_NAME, CSV_NAME)
    print('######### Dataset class created #########')
    print('Number of images: ', len(dataset))

    # BUILD TRAIN-VALIDATION-TEST SUBSETS
    train_loader, validation_loader, test_loader = build_train_valid_test_subsets(dataset)

    # CREATE NETWORK
    net = Net()
    net.cuda()
    print('######### Network created #########')
    print('Architecture:\n', net)

    # TRAIN
    train(train_loader, net)


def build_train_valid_test_subsets(dataset):
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

    return data_loader(dataset, train_indices), data_loader(dataset, val_indices), data_loader(dataset, test_indices)


def data_loader(dataset, indices):
    # Creating PT data samplers and loaders:
    sampler = SubsetRandomSampler(indices)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)


def train(train_loader, net):
    criterion = nn.CrossEntropyLoss()
    # SGD
    # optimizer = optim.Adam(net.parameters(), lr=0.0001) # 0.0001 - 0.001 - 0.01 - 0.1 - 0.00001
    optimizer = optim.Adadelta(net.parameters(), lr=1.0)  # 0.8 - 0.6 - 0.4
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

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


def validation():
    return
    # if epoch % 5 = 0:
    #     acc
    #     on
    #     val


if __name__ == '__main__':
    main()
