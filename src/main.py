from data_loader import load_data

data = load_data("perceptron-ANN/data/placement.csv")
# print(data.head())

dataset = load_data("perceptron-ANN/data/Churn_Modelling.csv")
dataset = dataset.drop(columns = ["RowNumber","Gender","Surname","Geography","CustomerId"],axis=0)
print(dataset.head())