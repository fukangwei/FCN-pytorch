from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
import torch

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def onehot(data, n):
    buf = np.zeros(data.shape + (n,))
    all_one = np.ones(data.shape)
    buf[:, :, 0] = data
    buf[:, :, 1] = all_one - data
    return buf


class BagDataset(Dataset):
    def __init__(self, transform=None, train=True):
        self.train = train
        self.transform = transform

    def __len__(self):
        if self.train is True:
            return len(os.listdir('./Dataset/train/'))
        else:
            return len(os.listdir('./Dataset/test/'))

    def __getitem__(self, idx):
        if self.train is True:
            img_name = os.listdir('./Dataset/train/')[idx]
            imgA = cv2.imread('./Dataset/train/' + img_name)
            imgA = cv2.resize(imgA, (160, 160))
            imgB = cv2.imread('./Dataset/train_mask/' + img_name, 0)
            imgB = cv2.resize(imgB, (160, 160))
            imgB = imgB / 255
            imgB = imgB.astype('uint8')
            imgB = onehot(imgB, 2)
            imgB = imgB.swapaxes(0, 2).swapaxes(1, 2)
            imgB = torch.FloatTensor(imgB)

            if self.transform:
                imgA = self.transform(imgA)
            item = {'A': imgA, 'B': imgB}
            return item
        else:
            img_name = os.listdir('./Dataset/test/')[idx]
            imgA = cv2.imread('./Dataset/test/' + img_name)
            imgA = cv2.resize(imgA, (160, 160))
            imgB = cv2.imread('./Dataset/test_mask/' + img_name, 0)
            imgB = cv2.resize(imgB, (160, 160))
            imgB = imgB / 255
            imgB = imgB.astype('uint8')
            imgB = onehot(imgB, 2)
            imgB = imgB.swapaxes(0, 2).swapaxes(1, 2)
            imgB = torch.FloatTensor(imgB)

            if self.transform:
                imgA = self.transform(imgA)
            item = {'A': imgA, 'B': imgB}
            return item


train_bag = BagDataset(transform, train=True)
train_dataloader = DataLoader(train_bag, batch_size=4, shuffle=True, num_workers=4)

test_bag = BagDataset(transform, train=False)
test_dataloader = DataLoader(test_bag, batch_size=4, shuffle=True, num_workers=4)
