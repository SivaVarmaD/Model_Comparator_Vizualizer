import streamlit as st
import pandas as pd
import pickle

# Load the models
models = {}
model_files = {
    'Linear Regression': 'models/linear_regression_model.pkl',
    'Support Vector Regression': 'models/svr_model.pkl',
    'Decision Tree': 'models/decision_tree_model.pkl',
    'Random Forest': 'models/random_forest_model.pkl',
    'Gradient Boosting': 'models/gradient_boosting_model.pkl'
}

for model_name, model_file in model_files.items():
    try:
        with open(model_file, 'rb') as file:
            models[model_name] = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading {model_name} model: {e}")

# Function to convert words to integer values
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'zero': 0}
    if isinstance(word, str):
        return word_dict.get(word.lower(), 0)
    elif isinstance(word, int):
        return word
    else:
        return 0

# Streamlit UI
st.title('Employee Salary Prediction')

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())
        
        # Preprocessing
        data['experience'] = data['experience'].fillna('zero')
        data['test_score'] = data['test_score'].fillna(data['test_score'].mean())
        data['experience'] = data['experience'].apply(lambda x: convert_to_int(x))

        st.write("Preprocessed Data:")
        st.write(data.head())

        experience = st.selectbox('Choose experience level', data['experience'].unique())
        test_score = st.number_input('Test Score', min_value=0, max_value=10, value=5)
        interview_score = st.number_input('Interview Score', min_value=0, max_value=10, value=5)
        
        model_type = st.selectbox('Choose model type', list(models.keys()))

        if st.button('Predict'):
            model_data = models.get(model_type)
            if model_data is None:
                st.error('Selected model not loaded. Please check the server.')
            else:
                try:
                    model = model_data['model']
                    scaler = model_data.get('scaler')  # Scaler might be None for some models

                    # Prepare input data
                    input_data = pd.DataFrame([[experience, test_score, interview_score]], columns=['experience', 'test_score', 'interview_score'])

                    # Scale features if needed
                    if scaler:
                        input_data = scaler.transform(input_data)

                    # Make prediction
                    prediction = model.predict(input_data)
                    output = round(prediction[0], 2)
                    st.success(f'Predicted Salary: ${output}')
                except Exception as e:
                    st.error(f'Error making prediction: {e}')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
else:
    st.write("Please upload a CSV file to get started.")
