from random import shuffle
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import os
import matplotlib.pyplot as plt 
torch.manual_seed(0)
np.random.seed(0)

class Solver(object):
    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss):
        self.optim_args = optim_args
        self.optim = optim
        self.loss_func = loss_func().cuda(0)

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def _log_results(self, train_loss,  val_loss, Log_dir, plotname, i ):
        if not os.path.exists(Log_dir):
            os.mkdir(Log_dir)    
        path = os.path.join(Log_dir, plotname + '_summary.txt')
        Error_file  = open(path, "w")
        Error_file.write('Train_loss: ' + str(train_loss))
        Error_file.write('\n val_loss: '+  str(val_loss))
        Error_file.close()

        plot_path = os.path.join(Log_dir, plotname + '.png')
        plt.figure(i+10, figsize=(15, 10))
        plt.plot(range(len(train_loss)), train_loss, 'r', label = 'train')
        plt.plot(range(len(val_loss)), val_loss, 'g', label = 'val')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Train & Validation ' + plotname)
        plt.savefig(plot_path)
        plt.close()

    def train(self, model, train_loader, val_loader, output_dir,patience = 5,  num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        - output_dir: Output directory for saving model and training plots
        - patience: For early stopping of the training

        Outputs:
        - training acc : Accuracy of the model on training dataset
        - best_val_acc : best Accuracy of the model on validation set
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda1)

        print('START TRAIN.')

        train_acc = 0
        best_val_acc = 0
        patience_count = 0
        for epoch in range(num_epochs):

            train_size = 0
            correct_train = 0
            train_loss = 0        
            for i, data in enumerate(train_loader, 0):
                # get the inputs, wrap them in Variable
                input_, label = data
                inputs, labels = Variable(input_).cuda(0), Variable(label).cuda(0)
                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                labels = labels.type(torch.long)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optim.step()

                # Storing values
                if (i+1) % log_nth == 0:
                    print(('[Iteration %d/%d] Train loss: %0.4f') % \
                          (i, iter_per_epoch, loss.item()))
                
                _,predicted_train = torch.max(outputs.data, 1)
                train_size += labels.size(0)
                correct_train += (predicted_train == labels).sum()
                train_loss += loss.item()
            
            self.train_loss_history.append(train_loss/float(train_size))            
            self.train_acc_history.append(float(correct_train)/float(train_size))
            
            ## Validation 
            correct_val = 0
            val_size = 0
            loss_val = 0
            for i, data_val in enumerate(val_loader, 0):
                # get the inputs, wrap them in Variable
                input_val, label_val = data_val
                inputs_val, labels_val = Variable(input_val).cuda(0), Variable(label_val).cuda(0)
                labels_val = labels_val.type(torch.long)
                output_val = model(inputs_val)
                loss = self.loss_func(output_val, labels_val)
                _,predicted_val = torch.max(output_val.data, 1)
                val_size += label_val.size(0)
                loss_val += loss
                correct_val += (predicted_val == labels_val).sum()
            self.val_acc_history.append(float(correct_val)/float(val_size))
            self.val_loss_history.append(loss_val.item()/float(val_size))  
            
            
            print(('[Epoch %d/%d] |     Train acc/loss: %0.4f/%0.4f   |     Val acc/loss: %0.4f/%0.4f') % \
                  (epoch, num_epochs, self.train_acc_history[-1], self.train_loss_history[-1], self.val_acc_history[-1], self.val_loss_history[-1]))

            if best_val_acc < self.val_acc_history[-1]:
                best_val_acc = self.val_acc_history[-1]
                train_acc = self.train_acc_history[-1]
                model.save(os.path.join(output_dir, 'best_model.pth'))
                patience_count = 0
            else:
                patience_count += 1

        
            self._log_results(self.train_loss_history, self.val_loss_history, output_dir, 'cross_entropy_loss', 1)
            self._log_results(self.train_acc_history, self.val_acc_history, output_dir, 'accuracy_plot', 2)
            
            if patience == patience_count:
                model.save(os.path.join(output_dir, 'model_early_stop' + str(num_epochs) + '.pth'))
                break
            scheduler.step()

        model.save(os.path.join(output_dir, 'model_' + str(num_epochs) + '.pth'))

        print('TRAINING FINISHED !')

        return train_acc, best_val_acc