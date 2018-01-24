from flask import Flask
from flask import request
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

#Examine if people should do certain activities under some intrinsic and extrinsic conditions 

app = Flask(__name__)

data = [ [] for i in range(5)]
lr = LinearRegression()
le1 = LabelEncoder()
le2 = LabelEncoder()

#only support entry-wise input
@app.route('/learn',methods=['GET', 'POST'])
def learn():
  raw_data = request.get_json()
  data[0].append(raw_data["activity"])
  data[1].append(raw_data["gender"])
  data[2].append(raw_data["temperature"])
  data[3].append(raw_data["humidity"])
  data[4].append(raw_data["score"])
  return "Input is stored"


@app.route('/train',methods=['GET', 'POST'])
def train():
  le1.fit(data[0])
  feature1 = le1.transform(data[0])
  le2.fit(data[1])
  feature2 = le2.transform(data[1])
  X = np.concatenate((feature1[:,np.newaxis],feature2[:,np.newaxis],np.array(data[2:4]).T),axis=1)
  y = np.array(data[4])
  lr.fit(X,y)
  return "Model Trained" 

#Not for new category features
@app.route('/predict',methods=['GET', 'POST'])  
def predict():  
  predict = request.get_json()   
  feature1=le1.transform([predict["activity"]])[0]
  feature2=le2.transform([predict["gender"]])[0]
  return "Predicted score: "+str(lr.predict(np.array([feature1, feature2, predict["temperature"], predict["humidity"] ])[np.newaxis,:])[0])
