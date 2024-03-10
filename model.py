import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

dataset = pd.read_csv('./Student_Performance.csv')

features = dataset[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']]
target = dataset[['Performance Index']]

training_x, testing_x, training_y, testing_y = train_test_split(features, target, test_size=0.2)

model = LinearRegression()
model.fit(training_x, training_y)

joblib.dump(model, 'model.pkl')

predictions = model.predict(testing_x)
error = mean_squared_error(testing_y, predictions)

print(f'error rate: {error}')