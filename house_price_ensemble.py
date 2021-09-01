#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:24:43 2021

@author: yinxiaoru
"""
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#载入训练和测试数据
train_data = pd.read_csv(os.path.abspath(sys.argv[1] + '/train.csv'),index_col = 0)
test_data = pd.read_csv(os.path.abspath(sys.argv[1] + '/test.csv'), index_col = 0)

#查看数据
print(train_data.shape,test_data.shape)

#准备训练数据的label
train_y = np.log1p(train_data.pop('SalePrice'))
train_y = train_y.values.astype("int")

#把训练和测试数据连在一起方便做数据预处理
all_x = pd.concat((train_data,test_data),axis = 0)
all_x.isnull().sum()

#对于内容项目进行one-hot-coding
all_dummy_data = pd.get_dummies(all_x)

#计算平均值来计算nan
mean_cols = all_dummy_data.mean()
all_dummy_data = all_dummy_data.fillna(mean_cols)
all_dummy_data.isnull().sum()

#对于数值列进行归一化处理
numerics_cols = all_x.columns[all_x.dtypes!="object"]
numerics_clos_means = all_dummy_data.loc[:,numerics_cols].mean()
numerics_cols_std = all_dummy_data.loc[:,numerics_cols].std()
all_dummy_data.loc[:,numerics_cols]=(all_dummy_data.loc[:,numerics_cols]-numerics_clos_means)/numerics_cols_std

#获得numpy 数组的格式用于训练
train_x = all_dummy_data.loc[train_data.index].values
test_x = all_dummy_data.loc[test_data.index].values

#搭建一个高级的ensemble model
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

#设定并训练模型
ridge=Ridge(alpha=15)
rf=RandomForestRegressor(n_estimators=500,max_feature=0.3)

ridge.fit(train_x,train_y)
rf.fit(train_x,trian_y)

#ensemble 预测
ridge_predict = np.expm1(ridge.predict(test_x))
rf_predict = np.expm1(rf.predict(test_x))

ensemble_predict = (ridge_predict + rf_predict)/2

submission_data = (pd.DataFrame({'Id':test_data.index,'SalePrice':ensemble_predict}))
submission_data.to_csv(os.path.abspath(sys.argv[2] + '/house_price_ensemble.csv'))
