#Another Regression model Lasso is used
from sklearn.datasets import load_diabetes
from sklearn.linear_model import  Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

diabetes=load_diabetes()
X=diabetes.data
y=diabetes.target

Lasso=Lasso()
parameters = {'alpha':np.linspace(-10,1,100)}
l = GridSearchCV(estimator=Lasso,param_grid=parameters)
l.fit(X,y)
print(l.best_score_)
#0.488658881723
print(l.best_params_)
#{'alpha': 0.0}

#Conclusion: The best parameter of alpha is 0.0, i.e ordinary linear regression, which is out of surprise.
#Conclusion: Lasso doesn't provide better prediction than ordinary linear regression, so we turn our focus on Elastic Net and LARS 
#Lasso
