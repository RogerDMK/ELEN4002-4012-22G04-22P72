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

def search_params(estimator, param_grid, search):
    """
    This is a helper function for tuning hyperparameters using teh two search methods.
    Methods must be GridSearchCV or RandomizedSearchCV.
    Inputs:
        estimator: Logistic regression, SVM, KNN, etc
        param_grid: Range of parameters to search
        search: Grid search or Randomized search
    Output:
        Returns the estimator instance, clf
    
    """   
    try:
        if search == "grid":
            clf = GridSearchCV(
                estimator=estimator, 
                param_grid=param_grid, 
                scoring=None,
                n_jobs=-1, 
                cv=10, 
                verbose=0,
                return_train_score=True
            )
        elif search == "random":           
            clf = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=10,
                n_jobs=-1,
                cv=10,
                verbose=0,
                random_state=1,
                return_train_score=True
            )
    except:
        print('Search argument has to be "grid" or "random"')
        
        
    # Fit the model
    clf.fit(X=X_train, y=Y_train.ravel())
    
    return clf

def black_box_function(C, degree):
    model = SVC(C = C, degree = degree)
    model.fit(X_train, Y_train.ravel())
    y_score = model.decision_function(X_test)
    f = roc_auc_score(Y_test.ravel(), y_score)
    return f


df = pd.read_csv('.\WildFires_DataSet.csv')
print(df.head())
print(df['CLASS'].value_counts())
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
# for i in runs:
#     clf = SVC(kernel='poly').fit(X_train,Y_train.ravel())
#     Pscoretrain = Pscoretrain + clf.score(X_train,Y_train)
#     Pscoretest  = Pscoretrain + clf.score(X_test,Y_test)
#     X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.30)

# print("Test")
# print(Pscoretest/len(runs))
# print("Train")
# print(Pscoretrain/len(runs))


svm_param = {
    "C": [.01, .1, 1, 5, 10, 100],
    "gamma": [0, .01, .1, 1, 5, 10, 100],
    "kernel": ["rbf"],
    "random_state": [1]
}

svm_dist = {
    "C": scipy.stats.expon(scale=.01),
    "gamma": scipy.stats.expon(scale=.01),
    "kernel": ["rbf"],
    "random_state": [1]
}
  
svm_grid = search_params(SVC(), svm_param, "grid")
acc = accuracy_score(y_true=Y_test, y_pred=svm_grid.predict(X_test))
print("**Grid search results**")
print("Best training accuracy:\t", svm_grid.best_score_)
print("Test accuracy:\t", acc)

svm_random = search_params(SVC(), svm_dist, "random")
acc = accuracy_score(y_true=Y_test, y_pred=svm_random.predict(X_test))
print("**Random search results**")
print("Best training accuracy:\t", svm_random.best_score_)
print("Test accuracy:\t", acc)

utility = UtilityFunction(kind = "ucb", kappa = 1.96, xi = 0.01)
optimizer = BayesianOptimization(f = None, 
                                 pbounds = {"C": [0.01, 10], 
                                            "degree": [1, 5]},
                                 verbose = 2, random_state = 1234)

for i in range(25):
    # Get optimizer to suggest new parameter values to try using the
    # specified acquisition function.
    next_point = optimizer.suggest(utility)  # Force degree from float to int.
    next_point["degree"] = int(next_point["degree"])    # Evaluate the output of the black_box_function using 
    # the new parameter values.
    target = black_box_function(**next_point)
    
    try:
        # Update the optimizer with the evaluation results. 
        # This should be in try-except to catch any errors!
        optimizer.register(params = next_point, target = target)
    except:
        pass
    
print("Best result: {}; f(x) = {:.3f}.".format(optimizer.max["params"], optimizer.max["target"]))