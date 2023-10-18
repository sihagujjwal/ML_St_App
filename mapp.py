import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Original columns after one hot encoding
original_columns = ['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc',
       'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',
       'revol_bal', 'revol_util', 'total_acc', 'mort_acc',
       'pub_rec_bankruptcies', 'zip_code_05113', 'zip_code_11650',
       'zip_code_22690', 'zip_code_29597', 'zip_code_30723', 'zip_code_48052',
       'zip_code_70466', 'zip_code_86630', 'zip_code_93700', 'sub_grade_A2',
       'sub_grade_A3', 'sub_grade_A4', 'sub_grade_A5', 'sub_grade_B1',
       'sub_grade_B2', 'sub_grade_B3', 'sub_grade_B4', 'sub_grade_B5',
       'sub_grade_C1', 'sub_grade_C2', 'sub_grade_C3', 'sub_grade_C4',
       'sub_grade_C5', 'sub_grade_D1', 'sub_grade_D2', 'sub_grade_D3',
       'sub_grade_D4', 'sub_grade_D5', 'sub_grade_E1', 'sub_grade_E2',
       'sub_grade_E3', 'sub_grade_E4', 'sub_grade_E5', 'sub_grade_F1',
       'sub_grade_F2', 'sub_grade_F3', 'sub_grade_F4', 'sub_grade_F5',
       'sub_grade_G1', 'sub_grade_G2', 'sub_grade_G3', 'sub_grade_G4',
       'sub_grade_G5', 'verification_status_Source Verified',
       'verification_status_Verified', 'purpose_credit_card',
       'purpose_debt_consolidation', 'purpose_educational',
       'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
       'purpose_medical', 'purpose_moving', 'purpose_other',
       'purpose_renewable_energy', 'purpose_small_business',
       'purpose_vacation', 'purpose_wedding', 'initial_list_status_w',
       'application_type_INDIVIDUAL', 'application_type_JOINT',
       'home_ownership_MORTGAGE', 'home_ownership_NONE',
       'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT']


# Load the saved model and preprocessors
model = tf.keras.models.load_model('llending_norm_model.h5')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')


def user_input_features():
    st.subheader("Loan Term")
    term = st.selectbox('Term', ['36', '60'])

    st.subheader("zip Code")
    zip_code = st.selectbox('Loan Grade', ['05113', '11650', '22690', '29597', '30723', '48052', '70466', '86630', '93700'])

    st.subheader("Loan SubGrade")
    sub_grade = st.selectbox('Loan Sub Grade', ['B3', 'A4', 'B5', 'B4', 'A1', 'B2', 'C1', 'A5', 'C2', 'A2', 'B1', 'A3', 'D2', 'C3', 'D3', 'C4', 'C5', 'D4', 'D5', 'D1', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2','F3', 'F4',
'F5', 'G1', 'G3', 'G4', 'G2', 'B3', 'A4', 'B5', 'B4', 'A1', 'B2', 'C1', 'A5', 'C2', 'A2', 'B1', 'A3', 'D2', 'C3', 'D3', 'C4', 'C5', 'D4', 'D5', 'D1', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1',
'G3', 'G4', 'G2', 'G5'])

#    st.subheader("Employement Length")
#    emp_length = st.selectbox('Employement Length', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    st.subheader("Public Record Bankrupcy")
    pub_rec_bankruptcies = st.selectbox('Public Record Bankrupcy', ['0', '1', '2', '3', '4', '5', '6', '7', '8'])

    st.subheader("Mortgage Account")
    mort_acc = st.selectbox('Mortgage Account', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'])

    st.subheader("Application Type")
    application_type = st.selectbox('Application Type', ['INDIVIDUAL', 'JOINT'])

    st.subheader("Home Ownership")
    home_ownership = st.selectbox('Home Ownership', ['RENT', 'MORTGAGE', 'OWN', 'NONE', 'OTHER'])

    st.subheader("Verification Status")
    verification_status = st.selectbox('Verification Status', ['Verified', 'Source Verified'])

#    st.subheader("Payment Plan")
#    pymnt_plan = st.selectbox('Payment Plan', ['y', 'n'])

    st.subheader("Initial List Status")
    initial_list_status = st.selectbox('Initial List Status', ['w']) 
                                       
    st.subheader("Purpose")
    purpose = st.selectbox('Purpose', ['debt_consolidation', 'credit_card', 'other', 'home_improvement', 'small_business', 'major_purchase', 'car', 'wedding', 'medical', 'moving', 'house', 'vacation', 'renewable_energy'])                      

    st.subheader("Enter below values in given textbox")
    loan_amnt = st.text_input("Enter loan amount in digits ")                                  
                                       
    int_rate = st.text_input("Enter interest rate in digits ")

    installment = st.text_input("Enter number of installments in digits ")

    annual_inc = st.text_input("Enter annual income in digits ")

    dti = st.text_input("Enter dti in digits ")

#    delinq_2yrs = st.text_input("Enter delinq 2yrs in digits ")

#    inq_last_6mths = st.text_input("Enter number of enquiry done in last 6 months in digits ")
                                       
    open_acc = st.text_input("Enter number of open accounts ")

    pub_rec = st.text_input("Enter if any public record ")
                                       
    revol_bal = st.text_input("Enter if revol balance")

    revol_util = st.text_input("Enter if revol util")

    total_acc = st.text_input("Enter total account balance in digits")

    last_pymnt_amnt = st.text_input("Enter last payment amount in digits")

    earliest_cr_line = st.text_input("Enter earlies cr line date")
                                       

    data = {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': installment,
        'annual_inc': annual_inc,
        'dti': dti,
        'earliest_cr_line': earliest_cr_line,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'mort_acc': mort_acc,
        'pub_rec_bankruptcies': pub_rec_bankruptcies,
        'zip_code': zip_code,
        'sub_grade': sub_grade,
        'verification_status': verification_status,
        'purpose': purpose,
        'initial_list_status': initial_list_status,
        'application_type': application_type,
        'home_ownership': home_ownership
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

