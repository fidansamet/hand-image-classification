import torch
import torchvision.transforms as transforms
import cv2
import pandas as pd
import numpy as np
from PIL import Image


transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
            sample = {}
            sample['img_path'] = self.root_dir + '/' + row[7]
            sample['img_class'] = self.get_class_idx(row[2], row[6])
            self.files.append(sample)

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]['img_path']
        img_class = self.files[idx]['img_class']
        image = Image.open(img_path).convert('RGB')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #one_hot_vector = np.zeros((8,), dtype=int)
        # one_hot_vector[img_class] = 1

        if self.transform:
            image = self.transform(image)

        # sample = {'image': image, 'class': torch.from_numpy(one_hot_vector)}
        sample = {'image': image, 'class': img_class}

        return sample

    def get_class_idx(self, gender, hand_aspect):
        split_aspects = hand_aspect.split()
        split_aspects.insert(0, gender)
        class_name = '-'.join(split_aspects)
        return self.CLASSES.index(class_name)
