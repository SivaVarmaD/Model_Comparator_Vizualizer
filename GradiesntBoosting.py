import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Creating and fitting the Gradient Boosting model
gradient_boosting_regressor = GradientBoostingRegressor()
gradient_boosting_regressor.fit(X_train, y_train)

# Saving the model and scaler
with open('gradient_boosting_model.pkl', 'wb') as file:
    pickle.dump({'model': gradient_boosting_regressor, 'scaler': scaler}, file)

print("Gradient Boosting model and scaler saved as 'gradient_boosting_model.pkl'")

# Loading and testing the model
try:
    with open('gradient_boosting_model.pkl', 'rb') as file:
        loaded = pickle.load(file)
    model = loaded['model']
    scaler = loaded['scaler']

    # Test features
    test_features = pd.DataFrame([[2, 9, 6]], columns=['experience', 'test_score', 'interview_score'])
    test_features['experience'] = test_features['experience'].apply(lambda x: convert_to_int(x))
    test_features_scaled = scaler.transform(test_features)

    # Making prediction
    prediction = model.predict(test_features_scaled)
    print(f"Prediction for [2, 9, 6]: {prediction}")
except FileNotFoundError:
    print("Error: The model file 'gradient_boosting_model.pkl' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
