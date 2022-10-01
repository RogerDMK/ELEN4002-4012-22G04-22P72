from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('.\WildFires_DataSet.csv')
print(df.head())
mapping = {'fire':1, 'no_fire':0}
df = df.replace({'CLASS': mapping})
df['LST'] = df['LST']/df['LST'].abs().max()
df['BURNED_AREA'] = df['BURNED_AREA']/df['BURNED_AREA'].abs().max()
X = df[['NDVI', 'LST','BURNED_AREA']].values
Y = df[['CLASS']].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.20)
runs = range(50)
Pscoretest = 0
Pscoretrain = 0
for i in runs:
    clf = SVC(kernel='poly').fit(X_train,Y_train.ravel())
    Pscoretrain = Pscoretrain + clf.score(X_train,Y_train)
    Pscoretest  = Pscoretrain + clf.score(X_test,Y_test)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.30)

print("Test")
print(Pscoretest/len(runs))
print("Train")
print(Pscoretrain/len(runs))