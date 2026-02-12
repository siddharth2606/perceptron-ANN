import joblib
import numpy as np
from keras.models import load_model

model = joblib.load("perceptron-ANN/src/perceptron.pkl")
prediction = model.predict([[7.09,28.0]])


model2 = load_model("perceptron-ANN/src/ANN.h5")
scaler = joblib.load("perceptron-ANN/src/ANN.pkl")
data = np.array([[699,39,1,0.00,2,0,0,93826.63]])
data = scaler.transform(data)
prediction2 = model2.predict(data)
print(prediction2)