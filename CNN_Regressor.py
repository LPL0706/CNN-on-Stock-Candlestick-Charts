#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 00:22:56 2022

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

from torchtools import EarlyStopping

from Classic_CNN_model import AlexNet

class myLoss(nn.Module): # 自定义的适用于股票收益预测的损失函数
    def __init__(self, mode='IC'):
        super(myLoss, self).__init__()
        self.mode = mode

    def forward(self, pred, label):
        if self.mode == 'IC':
            loss = -self.tensor_corr(pred, label)
        if self.mode == 'AdjMSE':
            pred = torch.squeeze(pred)
            beta = 2.5
            loss = (pred - label) ** 2
            adj_loss = beta - (beta - 0.5) / (1 + torch.exp(10000 * torch.mul(pred, label)))
            loss = beta * loss / (1 + adj_loss)
        return loss
    
    def tensor_corr(self, x, y): #计算2个tensor张量的person相关系数
        x, y = x.reshape(-1), y.reshape(-1)
        x_mean, y_mean = torch.mean(x), torch.mean(y)
        corr = (torch.sum((x - x_mean) * (y - y_mean))) / (torch.sqrt(torch.sum((x - x_mean) ** 2)) * torch.sqrt(torch.sum((y - y_mean) ** 2)))
        return corr  

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

# @torch.no_grad()是一种比forward()更方便的检验模型的计算方式
@torch.no_grad()
def validset_loss(model, valid_days): # 给出模型在验证集上的整体损失
    model.eval() # 固定模型参数
    # loss_func = nn.MSELoss()
    loss_func = myLoss(mode='IC')
    valid_loss = 0
    for date in valid_days: # 动态读取每一天的数据，避免占用内存过大
        X = np.load('/Users/liupeilin/Desktop/CNN/K_Line_Matrix/' + date + '/X.npy')
        y = pd.read_csv('/Users/liupeilin/Desktop/CNN/K_Line_Matrix/' + date + '/y.csv')
        feature  = list(X)
        label = list(np.array(y['Y2']).astype('float32'))
        
        valid_data = []
        for i in range(len(feature)):
            valid_data.append((feature[i], label[i]))
        valid_set = mySet(valid_data)
        valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False)
        
        for i, data in enumerate(valid_loader):
            X, y = data
            y_pred = model(X.to(torch.float32)).T[0]
            loss = loss_func(y_pred, y.to(torch.float))
            valid_loss += loss.item()
    valid_loss = valid_loss / len(valid_days)
    print('average loss on validset: ', valid_loss)
    return valid_loss
        
@torch.no_grad()
def model_backtest(model, all_stock_pool, train_days, test_days): # 给出模型在训练集和测试集上的IC和RankIC，同时记录测试集上的alpha表
    model.eval() # 固定模型参数，开始测试模式
    train_IC_list = []
    train_rank_IC_list = []
    test_IC_list = []
    test_rank_IC_list = []
    alpha = all_stock_pool.copy()   
    for date in train_days + test_days:
        X = np.load('/Users/liupeilin/Desktop/CNN/K_Line_Matrix/' + date + '/X.npy')
        y = pd.read_csv('/Users/liupeilin/Desktop/CNN/K_Line_Matrix/' + date + '/y.csv')
        feature  = list(X)
        label = list(np.array(y['Y2']).astype('float32'))
        stock_pool = list(y['Uid'])

        data = []
        for i in range(len(feature)):
            data.append((feature[i], label[i]))
        data_set = mySet(data)
        data_loader = DataLoader(data_set, batch_size=len(data_set), shuffle=False)

        for i, data in enumerate(data_loader):
            X, y = data
            y_pred = model(X.to(torch.float32)).T[0].detach().numpy()
    
        # 给出每一个交易日的IC和rank_IC
        y = pd.Series(y)
        y_rank = y.rank()
        y_pred = pd.Series(y_pred)
        y_pred_rank = y_pred.rank()
        IC = y_pred.corr(y)
        RankIC = y_pred_rank.corr(y_rank)
        if date <= train_days[-1]: # 训练集
            train_IC_list.append(IC)
            train_rank_IC_list.append(RankIC)
        else: # 测试集，同时记录alpha表
            test_IC_list.append(IC)
            test_rank_IC_list.append(RankIC)            
            pred_df = pd.DataFrame({'Uid': stock_pool, date: y_pred}, index=range(len(y_pred)))             
            alpha = pd.merge(alpha, pred_df, left_on='Uid', right_on='Uid', how='outer')
            alpha.sort_values('Uid', inplace=True)
    alpha.to_csv('/Users/liupeilin/Desktop/CNN/AlexNet/alpha.csv')
    return train_IC_list, train_rank_IC_list, test_IC_list, test_rank_IC_list

def model_train(train_days, valid_days, max_epoch=10, lr=0.0001, patience=2):  
    model = AlexNet(num_classes=1)
    # model = torch.load('/Users/liupeilin/Desktop/CNN/Alex_model.pkl')
    # loss_func = nn.MSELoss()
    loss_func = myLoss(mode='IC')
    optimizer = Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(max_epoch):
        epoch_loss = 0
        for date in train_days:
            day_loss = 0
            X = np.load('/Users/liupeilin/Desktop/CNN/K_Line_Matrix/' + date + '/X.npy')
            y = pd.read_csv('/Users/liupeilin/Desktop/CNN/K_Line_Matrix/' + date + '/y.csv')
            feature  = list(X)
            label = list(np.array(y['Y2']).astype('float32'))

            train_data = []
            for i in range(len(feature)):
                train_data.append((feature[i], label[i]))

            train_set = mySet(train_data)
            train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
            
            for i, data in enumerate(train_loader):
                optimizer.zero_grad() # 梯度清零
                X, y = data
                y_pred = model(X.to(torch.float32)).T[0]
                loss = loss_func(y_pred, y.to(torch.float))
                if pd.isnull(loss.item()):
                    continue
                loss.backward() # 反向传播
                optimizer.step()
                day_loss += loss.item()
            # print("Epoch %d, Date %s loss: %f"%(epoch + 1, date, day_loss))
            epoch_loss += day_loss
        train_loss_list.append(epoch_loss)
        valid_loss = validset_loss(model, valid_days)
        valid_loss_list.append(valid_loss)
        print("\n##### Epoch %d average loss: "%int(epoch + 1), epoch_loss, '#####\n')
        
        # 记录迭代过程中模型在训练集和验证集上的损失
        with open('回归问题训练日志.txt', 'a') as f:
            f.write('epoch_' + str(epoch + 1) + ', ' + str(epoch_loss) + ', ' + str(valid_loss) + '\n')
            f.close()
        
        # 早停，可以设置至少训练n个epoch，避免因前期验证集损失波动导致训练中止
        if epoch >= 4:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                with open('回归问题训练日志.txt', 'a') as f:
                    f.write('EarlyStopping\n')
                    now_time = datetime.datetime.now()
                    f.write(str(now_time))
                    f.close()
                break
            
    plt.subplot(121)
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.subplot(122)
    plt.plot(range(len(valid_loss_list)), valid_loss_list)
    
    return model, train_loss_list, valid_loss_list
    
def main():
    train_days = ['2015-03-02', '2015-04-01', '2015-05-04', '2015-06-01', '2015-07-01', '2015-08-03']
    valid_days = ['2015-09-01', '2015-10-08']
    test_days = ['2015-11-02', '2015-12-01']
    all_stock_pool = pd.read_csv('/Users/liupeilin/Desktop/股票量价数据/all_stock_pool.csv')
    
    # model = torch.load('/Users/liupeilin/Desktop/CNN/Alex_model.pkl')
    time1 = time()
    model, train_loss_list, valid_loss_list = model_train(train_days, valid_days, max_epoch=100, lr=0.0001, patience=2)
    time2 = time()
    print('训练用时：', time2 - time1, 's')
    
    train_IC_list, train_rank_IC_list, test_IC_list, test_rank_IC_list = model_backtest(model, all_stock_pool, train_days, test_days)
    train_IC_avg = np.mean(train_IC_list)
    train_rank_IC_avg = np.mean(train_rank_IC_list)
    train_win_rate = len([i for i in train_IC_list if i > 0]) / len(train_IC_list)
    print('训练集：均值IC=', train_IC_avg)
    print('训练集：均值RankIC=', train_rank_IC_avg)
    print('训练集：胜率=', train_win_rate)
       
    test_IC_avg = np.mean(test_IC_list)
    test_rank_IC_avg = np.mean(test_rank_IC_list)
    test_win_rate = len([i for i in test_IC_list if i > 0]) / len(test_IC_list)
    print('测试集：均值IC=', test_IC_avg)
    print('测试集：均值RankIC=', test_rank_IC_avg)
    print('测试集：胜率=', test_win_rate)
    
if __name__ == '__main__':
    main()
    





