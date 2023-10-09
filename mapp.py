import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Original columns after one hot encoding
original_columns = ['loan_amnt', 'term', 'int_rate', 'installment', 'emp_length',
       'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
       'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'last_pymnt_amnt',
       'home_ownership_any', 'home_ownership_mortgage', 'home_ownership_none',
       'home_ownership_other', 'home_ownership_own', 'home_ownership_rent',
       'verification_status_not_verified',
       'verification_status_source_verified', 'verification_status_verified',
       'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation',
       'purpose_educational', 'purpose_home_improvement', 'purpose_house',
       'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
       'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
       'purpose_vacation', 'purpose_wedding', 'grade_a', 'grade_b', 'grade_c',
       'grade_d', 'grade_e', 'grade_f', 'grade_g', 'pymnt_plan_n',
       'pymnt_plan_y']


# Load the saved model and preprocessors
model = tf.keras.models.load_model('lending_model.h5')
scaler = joblib.load('mscaler.pkl')
encoder = joblib.load('mencoder.pkl')


def user_input_features():
    st.subheader("Loan Term")
    term = st.selectbox('Term', ['36', '60'])

    st.subheader("Loan Grade")
    grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])

#    st.subheader("Loan SubGrade")
#    sub_grade = st.selectbox('Loan Sub Grade', ['B3', 'A4', 'B5', 'B4', 'A1', 'B2', 'C1', 'A5', 'C2', 'A2', 'B1', 'A3', 'D2', 'C3', 'D3', 'C4', 'C5', 'D4', 'D5', 'D1', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2','F3', 'F4',
#'F5', 'G1', 'G3', 'G4', 'G2', 'B3', 'A4', 'B5', 'B4', 'A1', 'B2', 'C1', 'A5', 'C2', 'A2', 'B1', 'A3', 'D2', 'C3', 'D3', 'C4', 'C5', 'D4', 'D5', 'D1', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1',
#'G3', 'G4', 'G2', 'G5'])

    st.subheader("Employement length")
    emp_length = st.selectbox('emp length', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    st.subheader("Home Ownership")
    home_ownership = st.selectbox('Home Ownership', ['rent', 'mortgage', 'own'])

    st.subheader("Verification Status")
    verification_status = st.selectbox('Verification Status', ['Verified', 'Source Verified', 'Not Verified'])

    pymnt_plan = '9952'                                 
                                       
    st.subheader("Purpose")
    purpose = st.selectbox('Purpose', ['debt_consolidation', 'credit_card', 'other', 'home_improvement', 'small_business', 'major_purchase', 'car', 'wedding', 'medical', 'moving', 'house', 'vacation', 'renewable_energy'])                      

    st.subheader("Enter below values")
    loan_amnt = st.text_input("Enter loan amount in digits ")                                  
                                       
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
        'last_pymnt_amnt': last_pymnt_amnt
    }

    return pd.DataFrame(data, index=[0])

#if "user_data" not in st.session_state:
#    st.session_state.user_data = user_input_features()


def main():
    st.title('Lending Risk Prediction')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Lending Risk Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    data = user_input_features()
    if st.button("Predict"):
        user_data_encoded = pd.get_dummies(data, drop_first=True)
        missing_cols = set(original_columns) - set(user_data_encoded.columns)
        for col in missing_cols:
            user_data_encoded[col] = 0        
        user_data_encoded = user_data_encoded[original_columns]
        user_data_scaled = scaler.transform(user_data_encoded)
        prediction = model.predict(user_data_scaled)
        print(f"prediction is :{prediction}")
        result = encoder.inverse_transform([int(prediction[0])])

        st.write(f"The lending risk is predicted as {result[0]}.")
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()

