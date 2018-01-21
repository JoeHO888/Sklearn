#No noramlization and searching for best parameters
from sklearn.datasets import load_diabetes
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

diabetes=load_diabetes()
X=diabetes.data
y=diabetes.target

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3)
lr=LinearRegression()
lr.fit(X_train,y_train)
score=lr.score(X_test,y_test)
print(score)
#score=0.469910918618

plt.scatter(lr.predict(X_test),y_test)
plt.savefig("diabetes1.png")
plt.show()

