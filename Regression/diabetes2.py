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
lr=LinearRegression()
lr.fit(X_train,y_train)
score=lr.score(X_test,y_test)
print(score)
#score=0.491914879849

plt.scatter(lr.predict(X_test),y_test)
plt.savefig("diabetes2.png")
plt.show()


#Conclusion: Normalization does not make the whole model better.
#Reason: Features range from 0.2 to 0.2, normalization doesn't make a difference.
