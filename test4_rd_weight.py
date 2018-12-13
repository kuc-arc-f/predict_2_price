# -*- coding: utf-8 -*-
# テスト4, 体重、身長の重回帰分析。
#


import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# 機械学習モジュール
import sklearn

#
# 学習データ
wdata = pd.read_csv("dat_weight.csv" 
              ,names=("weight", "height","mid_lenght","top_lenth") )

#print(wdata.head() )
from sklearn.model_selection import train_test_split

# モデル
from sklearn import linear_model

# モデルのインスタンス
l_model = linear_model.LinearRegression()
 
# 説明変数に "xx" 以外を利用
X = wdata.drop("weight", axis=1)

print(X.shape )
#print(X[:10 ] )
#quit()
#print( type( X) )
#print(X[: 10 ] )

# 目的変数
Y = wdata["weight"]
# 学習データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25 ,random_state=0)
print(X_train.shape , y_train.shape  )
print(X_test.shape , y_test.shape  )
#print( type( X_test ) )
#quit()

# fit
clf = l_model.fit(X_train,y_train)
print("train:",clf.__class__.__name__ ,clf.score(X_train,y_train))
print("test:",clf.__class__.__name__ , clf.score(X_test,y_test))
 
# 偏回帰係数
print(pd.DataFrame({"Name":X.columns,
                    "Coefficients":clf.coef_}).sort_values(by='Coefficients') )
 
# 切片 
print(clf.intercept_)
#quit()

#predict
#tdat =X_test[1: 2]
tdat =X_test[0: 5 ]
#print(tdat )
pred = l_model.predict(tdat )
#print(pred.shape )
print(pred )
#print(pred[: 10])
quit()

d  = np.array(pred )
frame1 = DataFrame(d )
print(frame1.shape)
print(frame1.head() )
