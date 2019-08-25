import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np 
torch.manual_seed(0)
np.random.seed(0)

class ThreeLayerCNN(nn.Module):
    """
    A Three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - fc - dropout - relu - fc
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters=32, kernel_size=3,
                 stride=1, pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride: Stride for the convolution layer.
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ThreeLayerCNN, self).__init__()
        channels, height, width = input_dim
        self.first_layer = nn.Sequential(
            nn.Conv2d(channels, num_filters,kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(), nn.MaxPool2d(pool))

        input_size = int(num_filters*(height/pool)*(width/pool))

        self.second_layer = nn.Sequential(nn.Linear(input_size, hidden_dim), nn.Dropout(dropout), nn.ReLU() )
        self.third_layer = nn.Linear(hidden_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
        
    def forward(self, x):
        
        out = self.first_layer(x)
        out = out.view(out.size(0), -1)
        out = self.second_layer(out)
        out = self.third_layer(out)
        
        return out

    def save(self, path):
        
        Inputs:
        - path: path string
        
class EightLayerCNN(nn.Module):
    
    """
    Eight-layer convolutional network with the following architecture:
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters=32, kernel_size=3,
                 stride=1,  pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride: Stride for the convolution layer.
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(EightLayerCNN, self).__init__()
        channels, height, width = input_dim
        self.first_layer = nn.Sequential(
            nn.Conv2d(channels, num_filters,kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(),\
                            nn.Conv2d(num_filters, 32,kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(),\
                                nn.MaxPool2d(pool))

        self.second_layer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(),\
                nn.Conv2d(64, 64,kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(),\
                    nn.MaxPool2d(pool))
        
        self.third_layer = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(),\
            nn.Conv2d(128, 128,kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU())

        self.avgpool = nn.AvgPool2d(7,7)

        self.fourth_layer = nn.Sequential(nn.Linear(128, 256), nn.Dropout(dropout), nn.ReLU() )
        self.fifth_layer = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
        
    def forward(self, x):
        
        out = self.first_layer(x)
        out = self.second_layer(out)
        out = self.third_layer(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fourth_layer(out)
        out = self.fifth_layer(out)
        
        return out

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self.state_dict(), path)

class EightLayerCNN_FeatStack(nn.Module):
    """
    A eight layer convolutional network with feature stacking with the following architecture:
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters=32, kernel_size=3,
                 stride=1, pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride: Stride for the convolution layer.
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(EightLayerCNN_FeatStack, self).__init__()
        channels, height, width = input_dim
        self.first_layer = nn.Sequential(
                            nn.Conv2d(channels, 32 ,kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(),\
                            nn.Conv2d(32, 32 ,kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(),\
                                nn.MaxPool2d(pool))
        self.layer1 = nn.Sequential(nn.Conv2d(32,  32, 1, stride, int((kernel_size-1)/2)), nn.ReLU(),\
                                nn.AvgPool2d(14,14))


        self.second_layer = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(),\
                nn.Conv2d(32, 32,kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(),\
                    nn.MaxPool2d(pool))
        
        self.layer2 = nn.Sequential(nn.Conv2d(32,  32, 1, stride, int((kernel_size-1)/2)), nn.ReLU(),\
                                nn.AvgPool2d(7,7))

        self.third_layer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU(),\
            nn.Conv2d(64, 64,kernel_size, stride, int((kernel_size-1)/2)), nn.ReLU())

        self.avgpool = nn.AvgPool2d(7,7)

        self.fourth_layer = nn.Sequential(nn.Linear(128, 64), nn.Dropout(dropout), nn.ReLU() )
        self.fifth_layer = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, mean=1, std=0.02)
                init.constant_(m.bias.data, 0)
        


    def forward(self, x):
        out = self.first_layer(x)
        out1 = self.layer1(out)
        out1 = out1.view(out1.size(0),-1)
        out = self.second_layer(out)
        out2 = self.layer2(out)
        out2 = out1.view(out2.size(0),-1)
        out = self.third_layer(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        out = torch.cat([out1, out2, out], dim=1)
        out = self.fourth_layer(out)
        out = self.fifth_layer(out)

        return out

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self.state_dict(), path)
