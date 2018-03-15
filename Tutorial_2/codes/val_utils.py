import gc
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import torch
import os
from sklearn.metrics import accuracy_score, roc_auc_score

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def validate_epoch(net, val_loader, criterion):
    probs_lst = []
    ground_truth = []
    net.eval()

    running_loss = 0.0
    n_batches = len(val_loader)
    sm = nn.Sigmoid()
    
    for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        
        labels = Variable(sample['label'].float().cuda(), volatile=True)
        inputs = Variable(sample['img'].cuda(), volatile=True)
        
        outputs = net(inputs).squeeze()
        loss = criterion(outputs, labels)

        targets = sample['label'].numpy()        
        preds = sm(outputs).data.cpu().numpy()

        probs_lst.append(preds) 
        ground_truth.append(targets)

        running_loss += loss.data[0]
        gc.collect()

    gc.collect()

    probs_lst = np.hstack(probs_lst)
    ground_truth = np.hstack(ground_truth)

    
    return running_loss/n_batches, roc_auc_score(ground_truth, probs_lst)
