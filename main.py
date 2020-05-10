import torch
import torch.optim as optim
import numpy as np
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from dataset import HandDataset
from network import device, Net
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# default `log_dir` is "runs" - we'll be more specific here
train_writer = SummaryWriter('runs/train/ADADELTA_batch=32_epoch=15_loss=nll_LR=1.0_train')
val_writer = SummaryWriter('runs/test/ADADELTA_batch=32_epoch=15_loss=nll_LR=1.0_test')


CSV_NAME = 'HandInfo.csv'
DATASET_NAME = 'Hands'
VALIDATION = .2
TEST = .2
SHUFFLE = True
BATCH_SIZE = 32
RANDOM_SEED = 42
EPOCH = 15
LR = 1.0


def main():
    # BUILD DATASET
    dataset = HandDataset(DATASET_NAME, CSV_NAME)
    print('######### Dataset class created #########')
    print('Number of images: ', len(dataset))

    # BUILD TRAIN-VALIDATION-TEST SUBSETS
    train_loader, validation_loader, test_loader = build_train_valid_test_subsets(dataset)

    dataiter = iter(train_loader)
    a = dataiter.next()
    images, labels = a['image'], a['class']

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    # train_writer.add_image('four_fashion_mnist_images', img_grid)

    # CREATE NETWORK
    net = Net()
    net.cuda()
    print('######### Network created #########')
    print('Architecture:\n', net)

    # TRAIN AND VALIDATE

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.0001) # 0.0001 - 0.001 - 0.01 - 0.1 - 0.00001
    optimizer = optim.Adadelta(net.parameters(), lr=LR)  # 0.8 - 0.6 - 0.4
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    val_acc, val_loss, _ = test(net, validation_loader, 0)
    train_acc, train_loss, _ = test(net, train_loader, 0)

    val_writer.add_scalar('loss', val_loss, 0)
    train_writer.add_scalar('loss', train_loss, 0)

    val_writer.add_scalar('acc', val_acc, 0)
    train_writer.add_scalar('acc', train_acc, 0)
    for epoch in range(1, EPOCH):
        train(net, train_loader, optimizer, epoch)

        val_acc, val_loss, _ = test(net, validation_loader, epoch)
        train_acc, train_loss, _ = test(net, train_loader, epoch)

        val_writer.add_scalar('loss', val_loss, epoch)
        train_writer.add_scalar('loss', train_loss, epoch)

        val_writer.add_scalar('acc', val_acc, epoch)
        train_writer.add_scalar('acc', train_acc, epoch)


        scheduler.step()

    print('Finished Training')



def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().clone().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# helper functions

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().clone().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            HandDataset.CLASSES[preds[idx]],
            probs[idx] * 100.0,
            HandDataset.CLASSES[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


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
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=6)


def train(net, train_loader, optimizer, epoch, running_loss = 0.0, examples = 0):
    net.train()
    for i, data in enumerate(train_loader, 0):
        # Get inputs and classes
        inputs, classes = data['image'].to(device), data['class'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # loss = criterion(outputs, classes)
        loss = F.nll_loss(outputs, classes)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.data
        examples += BATCH_SIZE
        print('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, running_loss / examples))





def test(net, validation_loader, epoch):
    net.eval()  # TODO
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(validation_loader, 0):
            inputs, classes = data['image'].to(device), data['class'].to(device)
            output = net(inputs)
            test_loss += F.nll_loss(output, classes, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(classes.view_as(pred)).sum().item()

    test_loss /= len(validation_loader.sampler.indices)
    test_acc = 100. * correct / len(validation_loader.sampler.indices)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader.sampler.indices),
        test_acc))

    return test_acc, test_loss, correct


# def draw(running_loss, epoch):
#     writer.add_scalar('training loss', running_loss, epoch)


if __name__ == '__main__':
    main()
