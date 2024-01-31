#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:35:29 2022

@author: liupeilin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from Classic_CNN_model import AlexNet
from torchtools import EarlyStopping

class mySet(Dataset):    
    def __init__(self, data):
        super(mySet, self).__init__()
        self.data = data
        
    def __getitem__(self, x):
        return self.data[x]
    
    def __len__(self):
        return len(self.data)
    
def save_checkpoint(model, optimizer, epoch, loss, path): # 保存模型参数与训练进度
    optim_state = optimizer.state_dict()
    checkpoint = {"model_state_dict" : model.state_dict(),
                  "epoch" : epoch,
                  "loss" : loss,
                  "optimizer_state_dict" : optim_state}
    torch.save(checkpoint, path)
        
def load_checkpoint(model, path, optimizer=Adam): # 加载模型参数与训练进度
    checkpoint = torch.load(path) 
    model.load_state_dict(checkpoint['model_state_dict'])
    if checkpoint['optimizer_state_dict'] is not None: 
        model.optimizer = optimizer(model.parameters()) 
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

@torch.no_grad()
def validset_loss(model, valid_days): # 给出模型在验证集上的平均损失
    model.eval() # 固定模型参数
    loss_func = nn.CrossEntropyLoss()
    valid_loss = 0
    for date in valid_days:
        X = np.load('/Users/liupeilin/Desktop/倪老师实习/CNN/K_Line_Matrix/' + date + '/X.npy')
        y = pd.read_csv('/Users/liupeilin/Desktop/倪老师实习/CNN/K_Line_Matrix/' + date + '/y.csv')
        feature  = list(X)
        label = list(y['Y2'])
        label = [1 if i >= 0 else 0 for i in label]
        
        valid_data = []
        for i in range(len(feature)):
            valid_data.append((feature[i], label[i]))
        valid_set = mySet(valid_data)
        valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False)
        
        for i, data in enumerate(valid_loader):
            X, y = data
            y_pred = model(X.to(torch.float32))
            loss = loss_func(y_pred, y.long())
            valid_loss += loss.item()
    valid_loss = valid_loss / len(valid_days)
    print('average loss on validset: ', valid_loss)
    return valid_loss

@torch.no_grad()
def evaluation(model, data_loader): # 给出模型预测的准确率与混淆矩阵
    model.eval()
    confusion_matrix = np.zeros((2, 2))
    numT = 0
    numF = 0
    for i, data in enumerate(data_loader):
        X, y = data
        output = model(X.to(torch.float32))
        _, y_pred = torch.max(output, axis=1)
        _T = torch.sum(y_pred == y).item()
        numT += _T
        numF += (len(y) - _T)
        for j in range(len(y)):
            confusion_matrix[y[j], y_pred[j]] += 1
    accuracy = numT / (numT + numF)
    return accuracy, confusion_matrix
        
@torch.no_grad()
def model_backtest(model, all_stock_pool, train_days, test_days): # 给出模型在训练集和测试集上的IC和RankIC，同时记录测试集上的alpha表
    model.eval() # 固定模型参数，开始测试模式
    train_accuracy_list = []
    train_confu_mt_list = []
    test_accuracy_list = []
    test_confu_mt_list = []
    for date in train_days + test_days:
        X = np.load('/Users/liupeilin/Desktop/倪老师实习/CNN/K_Line_Matrix/' + date + '/X.npy')
        y = pd.read_csv('/Users/liupeilin/Desktop/倪老师实习/CNN/K_Line_Matrix/' + date + '/y.csv')
        feature  = list(X)
        label = list(np.array(y['Y2']).astype('float32'))
        label = [1 if i >= 0 else 0 for i in label]
        
        data = []
        for i in range(len(feature)):
            data.append((feature[i], label[i]))
        data_set = mySet(data)
        data_loader = DataLoader(data_set, batch_size=len(data_set), shuffle=False)
        
        accuracy, confusion_matrix = evaluation(model, data_loader)
        # 给出每一个交易日的胜率
        if date <= train_days[-1]: # 训练集
            train_accuracy_list.append(accuracy)
            train_confu_mt_list.append(confusion_matrix)
        else: # 测试集，同时记录alpha表
            test_accuracy_list.append(accuracy)
            test_confu_mt_list.append(confusion_matrix)
    return train_accuracy_list, train_confu_mt_list, test_accuracy_list, test_confu_mt_list

def rebalance(feature, label, mode='sub'): # 处理正负样本不平衡的问题
    df = pd.DataFrame({'feature':feature, 'label':label})
    df_1 = df[df['label'] == 1]
    df_0 = df[df['label'] == 0]    
    if mode == 'sub':       
        if len(df_1) > len(df_0):
            index = np.random.randint(len(df_1), size=len(df_0))
            sub_df_1 = df_1.iloc[list(index)]
            sub_df = pd.concat([sub_df_1, df_0])
        elif len(df_1) < len(df_0):
            index = np.random.randint(len(df_0), size=len(df_1))
            sub_df_0 = df_0.iloc[list(index)]
            sub_df = pd.concat([sub_df_0, df_1])
        else:
            sub_df = df
        return list(sub_df['feature']), list(sub_df['label'])

def model_train(train_days, valid_days, max_epoch=10, lr=0.0001, patience=2):  
    model = AlexNet(num_classes=2)
    # model = torch.load('/Users/liupeilin/Desktop/CNN/Alex_model.pkl')
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    path = '/Users/liupeilin/Desktop/CNN/AlexNet'
    early_stopping = EarlyStopping(path=path, patience=patience)
    
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(max_epoch):
        epoch_loss = 0
        for date in train_days:
            day_loss = 0
            X = np.load('/Users/liupeilin/Desktop/倪老师实习/CNN/K_Line_Matrix/' + date + '/X.npy')
            y = pd.read_csv('/Users/liupeilin/Desktop/倪老师实习/CNN/K_Line_Matrix/' + date + '/y.csv')
            feature  = list(X)
            label = list(y['Y2'])
            label = [1 if i >= 0 else 0 for i in label]
            
            # 对不平衡的数据进行下采样（样本数足够多，不用上采样）
            feature, label = rebalance(feature, label, mode='sub')

            train_data = []
            for i in range(len(feature)):
                train_data.append((feature[i], label[i]))

            train_set = mySet(train_data)
            train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
            
            for i, data in enumerate(train_loader):
                optimizer.zero_grad() # 梯度清零
                X, y = data
                y_pred = model(X.to(torch.float32))
                loss = loss_func(y_pred, y.long())
                loss.backward() # 反向传播
                optimizer.step()
                day_loss += loss.item()
            print("Epoch %d, Date %s loss: %f"%(epoch + 1, date, day_loss))
            epoch_loss += day_loss
        epoch_loss /= len(train_days)
        train_loss_list.append(epoch_loss)
        valid_loss = validset_loss(model, valid_days)
        valid_loss_list.append(valid_loss)
        print("\n##### Epoch %d average loss: "%int(epoch + 1), epoch_loss, '#####\n')
        
        # 记录迭代过程中模型在训练集和验证集上的损失
        with open('分类问题训练日志.txt', 'a') as f: # 'a'代表逐行写入，不覆盖
            f.write('epoch_' + str(epoch + 1) + ', ' + str(epoch_loss) + ', ' + str(valid_loss) + '\n')
            f.close()
            
        # 早停，可以设置至少训练n个epoch，避免因前期验证集损失波动导致训练中止
        if epoch >= 4:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                with open('训练日志.txt', 'a') as f:
                    f.write('EarlyStopping\n')
                    now_time = datetime.datetime.now()
                    f.write(str(now_time))
                    f.close()
                break
    
    plt.subplot(121)
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.subplot(122)
    plt.plot(range(len(valid_loss_list)), valid_loss_list)
    
    torch.save(model, '/Users/liupeilin/Desktop/倪老师实习/CNN/AlexNet/model.pkl') # 保存模型
    return model, train_loss_list, valid_loss_list
    
def main():
    train_days = ['2015-03-02', '2015-04-01', '2015-05-04', '2015-06-01', '2015-07-01', '2015-08-03']
    valid_days = ['2015-09-01', '2015-10-08']
    test_days = ['2015-11-02', '2015-12-01']
    all_stock_pool = pd.read_csv('/Users/liupeilin/Desktop/股票数据/all_stock_pool.csv')
    
    time1 = time()
    model, train_loss_list, valid_loss_list = model_train(train_days, valid_days, max_epoch=20, lr=1e-5, patience=2)
    time2 = time()
    print('训练用时:', time2 - time1, 's')
    
    train_accuracy_list, train_confu_mt_list, test_accuracy_list, test_confu_mt_list = model_backtest(model, all_stock_pool, train_days, test_days)
    train_acc_avg = np.mean(train_accuracy_list)
    train_confu_mt = sum(train_confu_mt_list)
    print('训练集: 平均accuracy =', train_acc_avg)
    print('训练集: confusion matrix =\n', train_confu_mt)
       
    test_acc_avg = np.mean(test_accuracy_list)
    test_confu_mt = sum(test_confu_mt_list)
    print('测试集: 平均accuracy=', test_acc_avg)
    print('测试集: confusion matrix =\n', test_confu_mt)
    
if __name__ == '__main__':
    main()
    

