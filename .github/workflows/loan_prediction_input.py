# Simple Loan Approval Prediction for Phone
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample dataset (like dataset.csv)
data = {
    'Gender': ['Male','Female','Male','Male','Female','Male','Female','Female','Male','Male'],
    'Married': ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','Yes'],
    'Dependents': ['0','1','0','2','0','1','0','2','0','1'],
    'Education': ['Graduate','Not Graduate','Graduate','Graduate','Not Graduate','Graduate','Graduate','Not Graduate','Graduate','Graduate'],
    'Self_Employed': ['No','Yes','No','No','Yes','No','No','Yes','No','No'],
    'ApplicantIncome': [5000,3000,4000,6000,3500,4500,5500,2500,4000,6000],
    'CoapplicantIncome': [0,1500,0,0,1000,0,0,2000,0,0],
    'LoanAmount': [200,100,150,250,120,180,220,90,160,240],
    'Loan_Amount_Term': [360,120,360,360,180,360,360,360,360,360],
    'Credit_History': [1,0,1,1,0,1,1,0,1,1],
    'Property_Area': ['Urban','Rural','Urban','Semiurban','Rural','Urban','Urban','Rural','Urban','Semiurban'],
    'Loan_Status': ['Y','N','Y','Y','N','Y','Y','N','Y','Y']
}

df = pd.DataFrame(data)

# Convert text to numbers
le = LabelEncoder()
for col in ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status','Dependents']:
    df[col] = le.fit_transform(df[col])

# Split data
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show results
X_test['Predicted_Loan_Status'] = le.inverse_transform(y_pred)
X_test['Actual_Loan_Status'] = le.inverse_transform(y_test)

print("✅ Accuracy:", round(accuracy*100, 2), "%")
print("\nPredictions vs Actual:")
print(X_test[['Predicted_Loan_Status','Actual_Loan_Status']])
