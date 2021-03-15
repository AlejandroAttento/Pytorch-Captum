import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, auc

class simpleDenseNN(nn.Module):
  def __init__(self, in_size, hid_size, out_size, criterion = nn.BCELoss(), thh = 0.5, dpo_perc = 0.3):
    super().__init__()
    
    self.l1 = nn.Linear(in_size, hid_size)
    self.l2 = nn.Linear(hid_size, out_size)
    
    self.dpo = nn.Dropout(p=dpo_perc)

    self.criterion = criterion

    self.thh = thh
    
  def forward(self, x):
    z1 = self.l1(x)
    a1 = F.leaky_relu(z1)
    d1 = self.dpo(a1)
    
    z2 = self.l2(d1)
    a2 = torch.sigmoid(z2)
    return a2
  
  def __metrics(self, outputs, labels):
    preds = outputs.detach().numpy() > self.thh
    labels = np.vectorize(lambda x: False if x == 0 else True)(labels.numpy())
    
    f1_s = f1_score(labels.astype('int'), preds.astype('int'))
    precision_s = precision_score(labels.astype('int'), preds.astype('int'))
    recall_s = recall_score(labels.astype('int'), preds.astype('int'))
    accuracy_s = accuracy_score(labels.astype('int'), preds.astype('int'))
    
    return f1_s, precision_s, recall_s, accuracy_s
  
  def training_step(self, batch):
    features, labels = batch
        
    preds = self(features)
    loss = self.criterion(preds, labels)
    f1, prec, rec, acc = self.__metrics(preds, labels)
    return {'loss': loss, 'acc': acc, 'f1': f1, 'prec': prec, 'rec':rec}
    
  def testing_step(self, batch):
    features, labels = batch
        
    preds = self(features.float())
    loss = self.criterion(preds, labels)
    f1, prec, rec, acc = self.__metrics(preds, labels)
    return {'loss': loss, 'acc': acc, 'f1': f1, 'prec': prec, 'rec':rec}
  
  def validation_epoch_end(self, outputs):
    batch_losses = [ite['loss'] for ite in outputs]
    epoch_loss = torch.stack(batch_losses)
    
    batch_accs = [ite['acc'] for ite in outputs]
    epoch_acc = torch.stack(batch_accs)
    
    return {'loss': loss, 'acc': acc, 'f1': f1, 'prec': prec, 'rec':rec}