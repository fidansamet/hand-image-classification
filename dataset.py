import torch
import torchvision.transforms as transforms
import cv2
import pandas as pd
import numpy as np
from utils import Rescale, ToTensor


transform = transforms.Compose([
    Rescale(256),
    ToTensor(),
    # normalization
])


class HandDataset(torch.utils.data.Dataset):

    CLASSES = ('female-dorsal-right', 'female-dorsal-left', 'female-palmar-left', 'female-palmar-right',
               'male-dorsal-right', 'male-dorsal-left', 'male-palmar-left', 'male-palmar-right')

    def __init__(self, root_dir, csv_file, transform=transform):
        self.ground_truth = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.files = []

        for i, row in self.ground_truth.iterrows():
            o = {}
            o['img_path'] = self.root_dir + '/' + row[7]
            o['img_class'] = self.get_class_idx(row[2], row[6])
            self.files.append(o)

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]['img_path']
        img_class = self.files[idx]['img_class']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        one_hot_vector = np.zeros((8,), dtype=int)
        one_hot_vector[img_class] = 1

        o = {'image': image, 'class': one_hot_vector}

        if self.transform:
            o = self.transform(o)

        return o

    def get_class_idx(self, gender, hand_aspect):
        split_aspects = hand_aspect.split()
        split_aspects.insert(0, gender)
        class_name = '-'.join(split_aspects)
        return self.CLASSES.index(class_name)
