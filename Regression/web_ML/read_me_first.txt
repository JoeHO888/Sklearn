Project Idea:

  This project aims to measure the need of using Ambi Climate by a real-valued score under some scenarios, concerning activity user is doing, gender, temperature and humidity. The high is the score, users are more encouraged to turn on Ambi Climate.




Manual:

There are four stages to start this program.

1. Unzip the zip folder and type: 
   export FLASK_APP=project.py
   python -m flask run
   on unix command windows to initiatie this program

2. Open another unix command window on linux and use curl to post json format input to http://localhost:5000/learn for machine to store data.
   Example:  curl -H "Content-Type: application/json" -X POST -d '{"activity": "sleep", "gender": "M", "temperature": 23.7, "humidity": 0.5, "score": 3.1}' http://localhost:5000/learn
             curl -H "Content-Type: application/json" -X POST -d '{"activity": "Watch TV", "gender": "F", "temperature": 28.7, "humidity": 0.2, "score": 1.1}' http://localhost:5000/learn

3. type: curl -X POST http://localhost:5000/train to train the machine

4. Use curl to post input which you want to predict a score from to let machine predict a score for you.
   Example: curl -H "Content-Type: application/json"  -X POST -d '{"activity": "Watch TV", "gender": "F", "temperature": 21.7, "humidity": 0.9}' http://localhost:5000/predict



Tool used:

1. Numpy is used for data manipulation, because of its compatibility with Scikit learn and the advantages in data manipulation.

2. Sickit learn, a machine learning library is used for building model, training model and result prediction. Linear regression is adopted because of the real-valued output and high accuracy under few features of input condition.

3. Flask is used for building web service. Data input and manipulation rely on Flask as well.
