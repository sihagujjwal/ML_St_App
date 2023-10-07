import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Original columns after one hot encoding
original_columns = [
    'cap-shape_c', 'cap-shape_f', 'cap-shape_k', 'cap-shape_s', 'cap-shape_x',
    'cap-surface_g', 'cap-surface_s', 'cap-surface_y',
    'cap-color_c', 'cap-color_e', 'cap-color_g', 'cap-color_n', 'cap-color_p', 'cap-color_r', 'cap-color_u', 'cap-color_w', 'cap-color_y',
    'bruises_t',
    'odor_c', 'odor_f', 'odor_l', 'odor_m', 'odor_n', 'odor_p', 'odor_s', 'odor_y',
    'gill-attachment_f', 'gill-spacing_w', 'gill-size_n',
    'gill-color_e', 'gill-color_g', 'gill-color_h', 'gill-color_k', 'gill-color_n', 'gill-color_o', 'gill-color_p', 'gill-color_r', 'gill-color_u', 'gill-color_w', 'gill-color_y',
    'stalk-shape_t',
    'stalk-root_b', 'stalk-root_c', 'stalk-root_e', 'stalk-root_r',
    'stalk-surface-above-ring_k', 'stalk-surface-above-ring_s', 'stalk-surface-above-ring_y',
    'stalk-surface-below-ring_k', 'stalk-surface-below-ring_s', 'stalk-surface-below-ring_y',
    'stalk-color-above-ring_c', 'stalk-color-above-ring_e', 'stalk-color-above-ring_g', 'stalk-color-above-ring_n', 'stalk-color-above-ring_o', 'stalk-color-above-ring_p', 'stalk-color-above-ring_w', 'stalk-color-above-ring_y',
    'stalk-color-below-ring_c', 'stalk-color-below-ring_e', 'stalk-color-below-ring_g', 'stalk-color-below-ring_n', 'stalk-color-below-ring_o', 'stalk-color-below-ring_p', 'stalk-color-below-ring_w', 'stalk-color-below-ring_y',
    'veil-color_o', 'veil-color_w', 'veil-color_y',
    'ring-number_o', 'ring-number_t',
    'ring-type_f', 'ring-type_l', 'ring-type_n', 'ring-type_p',
    'spore-print-color_h', 'spore-print-color_k', 'spore-print-color_n', 'spore-print-color_o', 'spore-print-color_r', 'spore-print-color_u', 'spore-print-color_w', 'spore-print-color_y',
    'population_c', 'population_n', 'population_s', 'population_v', 'population_y',
    'habitat_g', 'habitat_l', 'habitat_m', 'habitat_p', 'habitat_u', 'habitat_w'
]


# Load the saved model and preprocessors
model = tf.keras.models.load_model('lending_model.h5')
scaler = joblib.load('mscaler.pkl')
encoder = joblib.load('mencoder.pkl')

st.title('Lending Risk Prediction')


def user_input_features():
    st.subheader("Loan Term")
    term = st.selectbox('Term', ['36 month', '60 month'])

    st.subheader("Loan Grade")
    grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])

    st.subheader("Loan SubGrade")
    sub_grade = st.selectbox('Loan Sub Grade', ['B3', 'A4', 'B5', 'B4', 'A1', 'B2', 'C1', 'A5', 'C2', 'A2', 'B1', 'A3', 'D2', 'C3', 'D3', 'C4', 'C5', 'D4', 'D5', 'D1', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2','F3', 'F4',
'F5', 'G1', 'G3', 'G4', 'G2', 'B3', 'A4', 'B5', 'B4', 'A1', 'B2', 'C1', 'A5', 'C2', 'A2', 'B1', 'A3', 'D2', 'C3', 'D3', 'C4', 'C5', 'D4', 'D5', 'D1', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1',
'G3', 'G4', 'G2', 'G5'])

    st.subheader("Employement length")
    emp_length = st.selectbox('emp length', ['10+ years', '2 years', '3 years', '< 1 year', '4 years', '5 years', '1 years', '6 years', '7 years', '8 years', '9 years'])

    st.subheader("Home Ownership")
    home_ownership = st.selectbox('Home Ownership', ['RENT', 'MORTGAGE', 'OWN'])

    st.subheader("Verification Status")
    verification_status = st.selectbox('Verification Status', ['Verified', 'Source Verified', 'Not Verified')

    pymnt_plan = '9952'                                 
                                       
    st.subheader("Purpose")
    purpose = st.selectbox('Purpose', ['debt_consolidation', 'credit_card', 'other', 'home_improvement', 'small_business', 'major_purchase', 'car', 'wedding', 'medical', 'moving', 'house', 'vacation', 'renewable_energy'                      

    loan_amnt = st.text_input("Enter loan amount in digits ")

    emp_title = st.text_input("Enter employee title ")                                   
                                       
    int_rate = st.text_input("Enter interest rate in digits ")

    installment = st.text_input("Enter number of installments in digits ")

    annual_inc = st.text_input("Enter annual income in digits ")

    dti = st.text_input("Enter dti in digits ")

    delinq_2yrs = st.text_input("Enter delinq 2yrs in digits ")

    inq_last_6mths = st.text_input("Enter number of enquiry done in last 6 months in digits ")
                                       
    open_acc = st.text_input("Enter number of open accounts ")

    pub_rec = st.text_input("Enter if any public record ")
                                       
    revol_bal = st.text_input("Enter if revol balance")

    revol_util = st.text_input("Enter if revol util")

    total_acc = st.text_input("Enter total account balance in digits")

    last_pymnt_amnt = st.text_input("Enter last payment amount in digits")
                                       

    data = {
        'term': term,
        'grade': grade,
        'sub_grade': sub_grade,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'verification_status': verification_status,
        'pymnt_plan': pymnt_plan,
        'purpose': purpose,
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'installment': installment,
        'annual_inc': annual_inc,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'inq_last_6mths': inq_last_6mths,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'last_pymnt_amnt': last_pymnt_amnt,
        'emp_title': emp_title
    }

    return pd.DataFrame(data, index=[0])

if "user_data" not in st.session_state:
    st.session_state.user_data = user_input_features()

if st.button("Predict"):
    user_data_encoded = pd.get_dummies(st.session_state.user_data, drop_first=True)
    user_data_encoded = user_data_encoded[original_columns]
    user_data_scaled = scaler.transform(user_data_encoded)

    prediction = model.predict(user_data_scaled)
    result = encoder.inverse_transform([int(prediction[0])])

    st.write(f"The mushroom is predicted to be {result[0]}.")

