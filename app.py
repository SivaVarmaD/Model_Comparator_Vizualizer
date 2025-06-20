import streamlit as st

st.set_page_config(page_title="Salary Predictor", layout="centered")  # <- FIRST LINE

import pandas as pd
import pickle
import os


# ------------------- Load Models ------------------- #
models = {}
model_files = {
    'Linear Regression': 'models/linear_regression_model.pkl',
    'Support Vector Regression': 'models/svr_model.pkl',
    'Decision Tree': 'models/decision_tree_model.pkl',
    'Random Forest': 'models/random_forest_model.pkl',
    'Gradient Boosting': 'models/gradient_boosting_model.pkl'
}

for model_name, model_path in model_files.items():
    try:
        with open(model_path, 'rb') as f:
            models[model_name] = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading {model_name} model: {e}")

# ------------------- Convert Experience ------------------- #
def convert_to_int(word):
    word_dict = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
        'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12
    }
    if isinstance(word, str):
        return word_dict.get(word.lower().strip(), 0)
    return int(word) if isinstance(word, (int, float)) else 0

# ------------------- Streamlit UI ------------------- #
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title('💼 Employee Salary Predictor')

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        # Handle missing data
        df['experience'] = df['experience'].fillna('zero').apply(convert_to_int)
        df['test_score'] = df['test_score'].fillna(df['test_score'].mean())

        st.subheader("✅ Preprocessed Data")
        st.dataframe(df)

        # Inputs
        experience = st.selectbox('👨‍💼 Experience (Years)', sorted(df['experience'].unique()))
        test_score = st.slider('📝 Test Score', 0, 10, 5)
        interview_score = st.slider('🎤 Interview Score', 0, 10, 5)

        model_type = st.selectbox('🤖 Choose Model', list(models.keys()))

        if st.button('🔮 Predict Salary'):
            model_bundle = models.get(model_type)
            if not model_bundle:
                st.error("Model not loaded properly.")
            else:
                try:
                    model = model_bundle['model']
                    scaler = model_bundle.get('scaler')  # optional

                    input_data = pd.DataFrame(
                        [[experience, test_score, interview_score]],
                        columns=['experience', 'test_score', 'interview_score']
                    )

                    if scaler:
                        input_data = scaler.transform(input_data)

                    prediction = model.predict(input_data)
                    st.success(f"💰 Predicted Salary: **${round(prediction[0], 2)}**")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
    except Exception as e:
        st.error(f"❗ Error reading file: {e}")
else:
    st.info("👆 Upload a CSV file to get started.")
