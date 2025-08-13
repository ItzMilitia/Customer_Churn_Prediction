# --- 1. Import The Necessary Libraries ---
import streamlit as st
import pandas as pd
import joblib
import os

# --- 2. Load The Saved Pipeline ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(APP_ROOT, 'artifacts')
pipeline = joblib.load(os.path.join(artifacts_dir, "final_pipeline.joblib"))

# --- 3. Set Up The Application UI ---
st.title('📉 Customer Churn Predictor')
st.markdown("""
This application uses a machine learning model to predict whether a customer is likely
to churn (cancel their subscription). Please enter the customer's details in the sidebar
to get a prediction.
""")

# --- Sidebar Input Widgets ---
# --- Sidebar Input Widgets ---
with st.sidebar:
    st.header('Customer Input Features')

    gender = st.selectbox('Gender', ['Male', 'Female'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=24, step=1)

    # New fields for age, city, and income
    age = st.number_input('Age', min_value=18, max_value=100, value=35, step=1)
    city = st.text_input('City', value='New York')
    income = st.number_input('Income ($)', min_value=0.0, value=50000.0, step=100.0)

    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check',
                                                     'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, value=50.0, step=0.01)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=1200.0, step=0.01)
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])

    input_data = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'age': age,                  # NEW
        'city': city,                # NEW
        'income': income,            # NEW
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

# --- 4. Prediction Function ---
def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])

    # feature engineering: total_services
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    input_df['total_services'] = (input_df[service_cols] == 'Yes').sum(axis=1)

    # pipeline handles preprocessing + prediction
    churn_proba = pipeline.predict_proba(input_df)[0][1]
    label = 'Churn' if churn_proba > 0.5 else 'No Churn'
    return label, churn_proba

# --- 5. Trigger Prediction and Show Results ---
if st.button('Predict Churn', key='predict_button'):
    prediction_label, churn_probability = make_prediction(input_data)

    st.subheader('Prediction Result')

    if prediction_label == 'Churn':
        st.error('Prediction: This customer is likely to CHURN.')
    else:
        st.success('Prediction: This customer is likely to NOT CHURN.')

    st.metric(label="Probability of Churn", value=f"{churn_probability:.2%}")
    st.progress(churn_probability)
