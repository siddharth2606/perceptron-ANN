import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from mlxtend.plotting import plot_decision_regions 
from data_loader import load_data

data = load_data("perceptron-ANN/data/placement.csv")
x = data.iloc[:,:-1]
y = data["placed"]

sns.scatterplot(x="cgpa",y="score",hue="placed",data=data)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import Perceptron
pt = Perceptron()
pt.fit(x_train,y_train)

joblib.dump(pt,"perceptron.pkl")

plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=pt)

# -------------------------------------------*********---------------------------------------



dataset = load_data("perceptron-ANN/data/Churn_Modelling.csv")
dataset = dataset.drop(columns = ["RowNumber","Gender","Surname","Geography","CustomerId"],axis=0)

x1 = dataset.iloc[:,:-1]
y1 = dataset["Exited"]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ss = StandardScaler()

x1_train,x1_test,y1_train,y1_test = train_test_split(x1,y1,test_size=0.2,random_state=42)
x1_train = ss.fit_transform(x1_train)
x1_test = ss.transform(x1_test)


import tensorflow
from keras.layers import Dense
from keras.models import Sequential

ann = Sequential()

ann.add(Dense(6,input_dim = 8,activation="relu"))
ann.add(Dense(4,activation="relu"))
ann.add(Dense(2,activation="relu"))
ann.add(Dense(1,activation="sigmoid"))

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
ann.fit(x1_train,y1_train,batch_size=100,epochs = 50)

ann.save("ANN.h5")
joblib.dump(ss,"ANN.pkl")