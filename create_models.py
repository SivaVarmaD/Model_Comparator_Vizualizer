import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load and preprocess the dataset
try:
    dataset = pd.read_csv('Deployment-flask-master\hiring.csv')
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

# Save Linear Regression model
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump({'model': linear_regressor}, file)

# SVR model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter tuning for SVR
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
grid_search = GridSearchCV(SVR(), param_grid, cv=5)
grid_search.fit(X_scaled, y)

# Save best SVR model and scaler
best_svr = grid_search.best_estimator_
with open('svr_model.pkl', 'wb') as file:
    pickle.dump({'model': best_svr, 'scaler': scaler}, file)

print("Models saved successfully.")
