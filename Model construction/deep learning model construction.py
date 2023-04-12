import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms,models

import os
import glob
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


path = r"C:\Users\admin\Desktop\gbc_clasification_revised\check_point"
train_path = r"C:\Users\admin\Desktop\gbc_clasification_revised\second_train_images"
train_imgs = glob.glob(train_path + '/*.png')


train_transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5)
                                      ,transforms.RandomVerticalFlip(p=0.5)
                                      #,transforms.ColorJitter(contrast=0.5)
                                      ,transforms.ToTensor()])


test_transform = transforms.Compose([transforms.ToTensor()])

# 创建训练集dataset
class train_custum_dataset(Dataset):
    def __init__(self,imgs,label):
        self.imgs = imgs
        self.label = label

    def __getitem__(self,idx):
        img = self.imgs[idx]
        img_pil = Image.open(img).convert('RGB')
        img_tensor = train_transform(img_pil)
        target = self.label[idx]
        return img_tensor,target
    def __len__(self):
        return len(self.imgs)


# 创建测试集dataset
class test_custum_dataset(Dataset):
    def __init__(self,imgs,label):
        self.imgs = imgs
        self.label = label

    def __getitem__(self,idx):
        img = self.imgs[idx]
        img_pil = Image.open(img).convert('RGB')
        img_tensor = test_transform(img_pil)
        target = self.label[idx]
        return img_tensor,target
    def __len__(self):
        return len(self.imgs)


train_dataset = train_custum_dataset(train_imgs,train_target)
train_loader = DataLoader(train_dataset,batch_size=15,shuffle=True)

test_dataset = test_custum_dataset(test_imgs,test_target)
test_loader = DataLoader(test_dataset,batch_size=15)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#采用resnet50模型
resnet50_ = models.resnet50()
resnet50_.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
net = resnet50_.to(device)


loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(),lr=0.0001)

epochs=100
for epoch in range(epochs):
    train_loss = 0
    train_correct = 0

    train_epoch_loss = []
    train_acc = []

    test_loss = 0
    test_correct = 0

    test_epoch_loss = []
    test_acc = []

    y_all = []
    outputs_all = []

    y_test_all = []
    outputs_test_all = []

    net.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        train_loss += loss.item()
        y_all.extend(y.tolist())
        outputs_all.extend(torch.max(pred, dim=1)[0].tolist())
        with torch.no_grad():
            train_correct += (pred.argmax(1) == y).sum().item()
    auc_train = roc_auc_score(y_all, outputs_all)

    pth = "Epoch{}.pth"
    torch.save(net.state_dict(), os.path.join(path, pth).format(epoch))
    train_epoch_loss.append(train_loss / len(train_loader.dataset))
    train_acc.append(train_correct / len(train_loader.dataset))

    print("Epoch{}:[train loss:{},train_auc:{},train_acc:{}]".format(epoch, train_loss / len(train_loader.dataset),
                                                                     auc_train, train_acc))
    net.eval()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = net(x)
        loss = loss_fn(pred, y)
        test_loss += loss.item()
        y_test_all.extend(y.tolist())
        outputs_test_all.extend(torch.max(pred, dim=1)[0].tolist())
        with torch.no_grad():
            test_correct += (pred.argmax(1) == y).sum().item()
    auc_test = roc_auc_score(y_test_all, outputs_test_all)

    test_epoch_loss.append(test_loss / len(test_loader.dataset))
    test_acc.append(test_correct / len(test_loader.dataset))
    print("Epoch{}:test loss:{},,auc_test:{}test_acc:{}".format(epoch, test_loss / len(test_loader.dataset), auc_test,
                                                                test_acc))




































