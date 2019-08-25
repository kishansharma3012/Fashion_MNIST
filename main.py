import sys, os, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from classifiers import models
from torch.autograd import Variable
import itertools
from classifiers.solver import Solver
from data_utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Arguments for training Fashion Mnist network.')
    
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--Model_num', dest='Model_num', help='Model number',
          default=0, type=int)    
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--wd', dest='wd', help='Weight rate.',
          default=0.001, type=float)
    parser.add_argument('--data_dir', dest='data_dir', help='Path to data_dir',
          default='', type=str)
    parser.add_argument('--output_dir', dest='output_dir', help='Path to output_dir',
          default='', type=str)
    parser.add_argument('--debug', dest='debug', help='debug',
          default='False', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_dir = os.path.join(os.getcwd(), args.data_dir)
    output_dir = os.path.join(os.getcwd(), args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## Loading dataset
    if args.debug == "True":
        train_data, val_data, test_data = get_Fashion_MNIST_data(data_dir, num_training=400, num_validation=120, num_test = 100,
                                                                            dtype=np.float32)
    else:
        train_data, val_data, test_data = get_Fashion_MNIST_data(data_dir, num_training=48000, num_validation=12000, num_test = 10000,
                                                                            dtype=np.float32)
    __spec__ = None
    print('Loading data.....................')
    print("Train size: %i" % len(train_data))
    print("Val size: %i" % len(val_data))
    print("Test size: %i" % len(test_data))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    best_val_acc = 0
    best_lr = 0
    best_wd = 0

    Learning_rates = [args.lr]
    Weight_decays = [args.wd]
    cudnn.enabled = True

    ## Hyperparameter search
    for lr in Learning_rates:
        for wd in Weight_decays:
            #model = models.Model3(hidden_dim=256, dropout = 0.2)

            if args.Model_num == 0:
                model = models.ThreeLayerCNN(hidden_dim=256, dropout = 0.2)
            elif args.Model_num == 1:
                model = models.EightLayerCNN(hidden_dim=256, dropout = 0.2)
            else:
                model = models.EightLayerCNN_FeatStack(hidden_dim=256, dropout = 0.2)
                
            model.cuda(0)

            optim_args = {"lr": lr,
                        "betas": (0.9, 0.999),
                        "eps": 1e-8,
                        "weight_decay": wd}
            solver = Solver(optim_args=optim_args)
            train_acc, val_acc = solver.train(model, train_loader, val_loader, output_dir, patience = 15, log_nth=2000, num_epochs=args.num_epochs)

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_lr = lr 
                best_wd = wd
                model.save(os.path.join(output_dir, 'best_model_hyperparameter.pth'))

    if best_val_acc == 0:
        best_val_acc = val_acc
    
    ## Testing best model
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    test_scores = []

    best_model_path = os.path.join(output_dir, 'best_model_hyperparameter.pth')
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(output_dir, 'model_' + str(args.num_epochs) + '.pth')
    
    saved_state_dict = torch.load(best_model_path)
    model.load_state_dict(saved_state_dict)

    for batch in test_loader:
        inputs, labels = Variable(batch[0]).cuda(0), Variable(batch[1]).cuda(0)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        labels = labels.type(torch.long)
        preds = preds.cpu()
        labels = labels.cpu()
        test_scores.extend((preds == labels).data.numpy())

    test_acc = np.mean(test_scores)

    print('Test set accuracy: %f' % np.mean(test_scores))
    Error_file  = open(os.path.join(output_dir, 'Log_File.txt'), "w")
    Error_file.write('Train_acc: ' + str(train_acc))
    Error_file.write('\nVal_acc: '+  str(best_val_acc))
    Error_file.write('\nTest_acc: '+  str(test_acc))
    Error_file.close()
