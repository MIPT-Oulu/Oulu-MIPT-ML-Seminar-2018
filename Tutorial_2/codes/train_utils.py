"""
Training utilities.

(c) Aleksei Tiulpin, University of Oulu, 2018

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import gc


def train_epoch(epoch, net, optimizer, train_loader, criterion, max_ep):

    net.train(True)

    running_loss = 0.0
    n_batches = len(train_loader)
    pbar = tqdm(total=n_batches)
    for i, sample in enumerate(train_loader):
        optimizer.zero_grad()
        # forward + backward + optimize
        labels = Variable(sample['label'].float())
        inputs = Variable(sample['img'], requires_grad=True)
        
        outputs = net(inputs).squeeze()

        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        pbar.set_description('Train loss: %.3f / loss %.3f' % (running_loss / (i+1), loss.data[0]))
        pbar.update()
        gc.collect()
    gc.collect()
    pbar.close()
    return running_loss/n_batches
