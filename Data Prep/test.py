from random import sample
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from bayes_opt import BayesianOptimization, UtilityFunction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
from tqdm import tqdm

inputData = pd.read_csv('.\input_old.csv')
outputData = pd.read_csv('.\output_old.csv')
inputData.drop(inputData.columns[inputData.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
outputData.drop(outputData.columns[outputData.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
count_fires = 0
joined_pd = pd.concat([inputData, outputData], axis=1)

no_fire_df = joined_pd[joined_pd['FIRE'] == 0]
fire_df = joined_pd[joined_pd['FIRE'] == 1]
sampled_df = no_fire_df.sample(2*fire_df.size)
df = pd.concat([sampled_df, fire_df])

df['DAY_LST'] = df['DAY_LST']/df['DAY_LST'].abs().max()
df['NDVI'] = df['NDVI']/df['NDVI'].abs().max()
df['NIGHT_LST'] = df['NIGHT_LST']/df['NIGHT_LST'].abs().max()
X = df[['NDVI', 'DAY_LST', 'NIGHT_LST']].values
Y = df[['FIRE']].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1)

filename = 'trained_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
y_score = loaded_model.decision_function(X)
f = roc_auc_score(Y.ravel(), y_score)
print(f)