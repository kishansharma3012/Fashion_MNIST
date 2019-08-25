import numpy as np
import os
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.datasets as dset
import random 
import math

torch.manual_seed(0)
np.random.seed(0)

class Fashion_MNIST_Data(data.Dataset):
    
    def __init__(self, X, y, transform = None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]
        img = Image.fromarray(img, mode='L')

        
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.y)


def get_Fashion_MNIST_data(data_dir, num_training=48000, num_validation=12000, num_test = 10000,
                     dtype=np.float32):
    """
    Load the Fashion_Mnist-10 dataset from disk and perform preprocessing to prepare
    it for classifiers.
    """
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        is_download = True
    else:
        is_download = False

    raw_train_data = dset.FashionMNIST(data_dir, train=True, transform=None, target_transform=None, download=is_download)
    raw_test_data = dset.FashionMNIST(data_dir, train=False, transform=None, target_transform=None, download=is_download)

    X = []
    Y = []
    for i in range(len(raw_train_data)):
        X.append(np.array(raw_train_data[i][0]))
        Y.append(raw_train_data[i][1])

    X = np.array(X)
    Y = np.array(Y)
    X_test = []
    Y_test = []
    for i in range(len(raw_test_data)):
        X_test.append(np.array(raw_test_data[i][0]))
        Y_test.append(raw_test_data[i][1])

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    # Subsample the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    indices_train = indices[:num_training]
    indices_val = indices[num_training : num_training + num_validation ]
    X_train = X[indices_train]
    y_train = Y[indices_train]

    X_val = X[indices_val]
    y_val = Y[indices_val]
    
    X_test = X_test[:num_test]
    y_test = Y_test[:num_test]


    augmentation_transforms = transforms.Compose([
                            #transforms.RandomCrop(28, padding=2),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),    
                            RandomErasing(probability = 0.2, mean = [0.4914]),
                            ])
    default_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        ])

    return (Fashion_MNIST_Data(X_train, y_train, transform = augmentation_transforms),
            Fashion_MNIST_Data(X_val, y_val, transform=default_transforms),
            Fashion_MNIST_Data(X_test, y_test, transform= default_transforms))

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img