import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load and preprocess the dataset
try:
    dataset = pd.read_csv('hiring.csv')
except FileNotFoundError:
    print("Error: The file 'hiring.csv' was not found.")
    exit()

# Handling missing values
dataset['experience'] = dataset['experience'].fillna('zero')
dataset['test_score'] = dataset['test_score'].fillna(dataset['test_score'].mean())

# Define features and target variable
X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

# Convert experience to integer
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'zero': 0}
    return word_dict.get(word, 0)

X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

# Linear Regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Save Linear Regression model and scaler
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump({'model': linear_regressor}, file)

# SVR model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
svr_regressor = SVR()
svr_regressor.fit(X_scaled, y)

# Save SVR model and scaler
with open('svr_model.pkl', 'wb') as file:
    pickle.dump({'model': svr_regressor, 'scaler': scaler}, file)

print("Models saved successfully.")
