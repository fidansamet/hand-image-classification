This program classifies hand images by using Convolutional Neural Networks. It reads all images on the dataset in the directory with given name, reads CSV file with given file name and classifes hand images in one of the 8 classes which are female-dorsal-right, female-dorsal-left, female-palmar-left, female-palmar-right, male-dorsal-right, male-dorsal-left, male-palmar-left and male-palmar-right.

There are 4 files to run the program: main.py, dataset.py, mynet.py and resnet18.py.

Loss and accuracy plots will be shown as outputs. Also console logs will show the state of the program while running.


To run the program run the following command:

python3 main.py



Classes & Functions:

HandDataset: This class is used for data loading. It reads images from dataset, determines their classes according to given labels in csv files, resizes, converts and normalize images so that they are ready for training.
MyNet: This class contains CNN classifier that modeled from scratch. It conatins layers of network and forward function that is explained below.
ResNet18: This class contains fine-tuned ResNet-18 network. It conatins gradient requires flag setter function and forward function that applies softmax after ResNet-18 network since NLL loss function is used on backpropagation.

train_model: This function is the core function of model training. It sets optimizer, scheduler and loops given number of epochs for training. For each epoch, trains and tests on validation set, calculates accuracies and losses. Finally, tests on test set and calculates test accuracy and loss.
build_train_valid_test_subsets: This function builds sets for train, validation and test. It uses HandDataset class to build dataset. Then shuffles the dataset according to given random seed. Then splits data into three sets and returns them.
data_loader: This function loads data according to given dataset and indices. Returns obtained data.
train: This function trains the model according to given data and calculates the loss.
test: This function tests the model according to given data and calculates test accuracy and loss. Finally, returns them.



Globals:

RESNET: Selects between trained model from scratch or fine-tuned ResNet-18 model.
CSV_NAME: Specifies the file name of the CSV file for hand dataset.
DATASET_NAME: Specifies the directory name of the dataset.
VALIDATION: Specifies the divide rate of the dataset for validation dataset.
TEST: Specifies the divide rate of the dataset for test dataset.
SHUFFLE: Specifies the shuffling will be used on data load or not.
BATCH_SIZE: Specifies the batch size.
RANDOM_SEED: Specifies the random seed.
EPOCH: Specifies the epoch number incremented by one.
LR: Specifies the learning rate for optimization.
DEVICE: Specifies the device where the program will run.


