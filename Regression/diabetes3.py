#If change in parameter can't enhance the model greatly, cross validation will be forewent.

from sklearn.datasets import load_diabetes
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn import preprocessing 

diabetes=load_diabetes()
X_raw=diabetes.data
y=diabetes.target

X_normalized=preprocessing.scale(X_raw)
X_train, X_test, y_train, y_test=train_test_split(X_normalized,y,test_size=0.3)
lr=LinearRegression(fit_intercept=False)
lr.fit(X_train,y_train)
score=lr.score(X_test,y_test)
print(score)
#score=-3.285289767

#Conclusion: In this example, it is unwise to not fit the intercept.
#Conclusion: Another model has to be chosen. Accoding the sklearn cheat sheet, Lasso and Elastic Net will be chosen.
#Reason: Aftering plotting graph of features, many of features seem irrealvant to the target by visulization from the below a 
#piece of code

#for i in range(10):
#    plt.scatter(X_raw[:,i], y)
#    plt.show()
