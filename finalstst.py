import streamlit as st
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


# Load the trained model from a pickle file
with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

model_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']


# Define a function to preprocess the user input data
def preprocess_data(df):
    # Replace missing values with 0
    df.fillna(0, inplace=True)
    # Drop any remaining rows with missing values
    df.dropna(inplace=True)
    
    # Transform categorical variables
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                'PaperlessBilling', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Make sure all categorical columns are included in one-hot encoding
    missing_cols = set(['gender_Male', 'SeniorCitizen_1']) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
    
    # Scale numerical variables
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Reorder columns to match model input
    df = df.reindex(columns=model_columns, fill_value=0)
    
    return df


# Define the columns for user input data
cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

# Create an empty DataFrame to store user input data
user_input = pd.DataFrame(columns=cols)

# Define the form inputs for user input data
gender = st.selectbox('Gender', ['Male', 'Female'])
user_input.loc[0, 'gender'] = gender

senior = st.selectbox('Senior Citizen', ['Yes', 'No'])
user_input.loc[0, 'SeniorCitizen'] = 1 if senior == 'Yes' else 0

partner = st.selectbox('Partner', ['Yes', 'No'])
user_input.loc[0, 'Partner'] = partner

dependents = st.selectbox('Dependents', ['Yes', 'No'])
user_input.loc[0, 'Dependents'] = dependents

tenure = st.slider('Tenure', min_value=0, max_value=72, value=0, step=1)
user_input.loc[0, 'tenure'] = tenure

phone = st.selectbox('Phone Service', ['Yes', 'No'])
user_input.loc[0, 'PhoneService'] = phone
if phone == 'Yes':
    lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
else:
    lines = 'No phone service'
user_input.loc[0, 'MultipleLines'] = lines

internet = st.selectbox('Internet Service', ['No', 'DSL', 'Fiber optic'])
user_input.loc[0, 'InternetService'] = internet

if internet != 'No':
    security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    user_input.loc[0, 'OnlineSecurity'] = security
    backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    user_input.loc[0, 'OnlineBackup'] = backup
    protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    user_input.loc[0, 'DeviceProtection'] = protection
    support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    user_input.loc[0, 'TechSupport'] = support
    tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    user_input.loc[0, 'StreamingTV'] = tv
    movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    user_input.loc[0, 'StreamingMovies'] = movies
else:
    user_input.loc[0, 'OnlineSecurity'] = 'No internet service'
    user_input.loc[0, 'OnlineBackup'] = 'No internet service'
    user_input.loc[0, 'DeviceProtection'] = 'No internet service'
    user_input.loc[0, 'TechSupport'] = 'No internet service'
    user_input.loc[0, 'StreamingTV'] = 'No internet service'
    user_input.loc[0, 'StreamingMovies'] = 'No internet service'

contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
user_input.loc[0, 'Contract'] = contract

billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
user_input.loc[0, 'PaperlessBilling'] = billing

payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
user_input.loc[0, 'PaymentMethod'] = payment

charges = st.slider('Monthly Charges', min_value=0, max_value=200, value=0, step=1)
user_input.loc[0, 'MonthlyCharges'] = charges

total_charges = st.slider('Total Charges', min_value=0, max_value=10000, value=0, step=1)
user_input.loc[0, 'TotalCharges'] = total_charges

# Preprocess the user input data
user_input = preprocess_data(user_input)

# Use the model to make predictions on the user input data
prediction = model.predict(user_input)

print(model)

# Display the prediction to the user
if prediction == 0:
    st.write('The customer is likely to stay.')
else:
    st.write('The customer is likely to churn.')

