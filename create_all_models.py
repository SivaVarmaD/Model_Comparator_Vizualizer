import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Importing the dataset
try:
    dataset = pd.read_csv('hiring.csv')
except FileNotFoundError:
    print("Error: The file 'hiring.csv' was not found.")
    exit()

# Handling missing values
dataset['experience'] = dataset['experience'].fillna('zero')
dataset['test_score'] = dataset['test_score'].fillna(dataset['test_score'].mean())

# Defining features and target variable
X = dataset[['experience', 'test_score', 'interview_score']]
y = dataset['salary']

# Converting words to integer values
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'zero': 0}
    return word_dict.get(word, 0)  # Default to 0 if word not found

X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# List of models to create and save
models = {
    'linear_regression_model.pkl': LinearRegression(),
    'svr_model.pkl': SVR(kernel='rbf'),
    'decision_tree_model.pkl': DecisionTreeRegressor(),
    'random_forest_model.pkl': RandomForestRegressor(),
    'gradient_boosting_model.pkl': GradientBoostingRegressor()
}

# Fit each model and save it with metrics
for model_file, model in models.items():
    print(f"Training {model_file}...")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plotting loss curves (only available for some models)
    plt.figure()
    plt.plot(y_test.values, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Salary')
    plt.title(f'{model_file} - True vs Predicted')
    plt.legend()
    plt.grid(True)
    loss_plot = plt.gcf()  # Get current figure for saving

    # Save the model and metrics
    with open(f'models/{model_file}', 'wb') as file:
        pickle.dump({'model': model, 'scaler': scaler, 'metrics': {'mse': mse, 'r2': r2, 'loss_plot': loss_plot}}, file)

    print(f"{model_file} saved as 'models/{model_file}'")

# Optional: Test the models
for model_file in models.keys():
    try:
        with open(f'models/{model_file}', 'rb') as file:
            loaded = pickle.load(file)
        model = loaded['model']
        scaler = loaded['scaler']

        # Test features
        test_features = pd.DataFrame([[2, 9, 6]], columns=['experience', 'test_score', 'interview_score'])
        test_features['experience'] = test_features['experience'].apply(lambda x: convert_to_int(x))
        test_features_scaled = scaler.transform(test_features)

        # Making prediction
        prediction = model.predict(test_features_scaled)
        print(f"Prediction for [2, 9, 6] using {model_file}: {prediction}")
    except FileNotFoundError:
        print(f"Error: The model file '{model_file}' was not found.")
    except Exception as e:
        print(f"An error occurred with {model_file}: {e}")
