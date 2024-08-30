import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def loadBatches(path, is_train=True):
    if is_train:
        df = pd.read_csv(path)
        le = LabelEncoder()
        for column in df.select_dtypes(include=["object"]).columns:
            df[column] = le.fit_transform(df[column])

        features = df[["Credit Score", "Income", "Employment Status",
                       "Debt-to-Income Ratio", "Loan Amount", "Loan Purpose", "Marital Status",
                       "Number of Dependents", "Residence Status", "Previous Loan History",
                       "Bank Account Balance", "Education Level", "Existing Loan Balance", "Maximum Loan Limit"]]
        label = df["Loan Approval Status"]

        train_data, valid_data, train_label, valid_label = train_test_split(features, label, test_size=0.3, random_state=42, shuffle=True)

        return (train_data, train_label), (valid_data, valid_label)

    else:
        df = pd.read_csv(path)
        le = LabelEncoder()
        for column in df.select_dtypes(include=["object"]).columns:
            df[column] = le.fit_transform(df[column])

        features = df[["Credit Score", "Income", "Employment Status",
                       "Debt-to-Income Ratio", "Loan Amount", "Loan Purpose", "Marital Status",
                       "Number of Dependents", "Residence Status", "Previous Loan History",
                       "Bank Account Balance", "Education Level", "Existing Loan Balance", "Maximum Loan Limit"]]
        label = df["Loan Approval Status"]

        return (features, label)