import numpy as np
import pandas as pd

# Train Data 
np.random.seed(42)
size = 10000
file_name = "loan_approval.csv"

# Test Data
# np.random.seed(41)
# size = 1000
# file_name = "loan_approval_test.csv"

# 고용 상태
employment_status = np.random.choice(["Employed", "Self-employed", "Unemployed"], size=size, p=[0.7, 0.2, 0.1])

# 연 소득
incomes = np.random.lognormal(mean=np.log(60000), sigma=0.5, size=size).astype(int)
incomes[employment_status == "Unemployed"] = 0

# DTI
dti_ratios = np.random.normal(0.2, 0.1, size=size).clip(0.1, 0.4).round(3)

# 대출 신청 금액
loan_amounts = np.random.lognormal(mean=np.log(15000), sigma=1, size=size).astype(int)

# 대출 목적
loan_purposes = np.random.choice(["Home Purchase", "Car Purchase", "Education", "Personal Expenses"], size=size, p=[0.3, 0.2, 0.2, 0.3])

# 결혼 여부
marital_status = np.random.choice(["Single", "Married", "Divorced"], size=size, p=[0.5, 0.45, 0.05])

# 부양 가족 수
dependents = []
for status in marital_status:
    if status == "Married":
        num = np.random.poisson(lam=1) + 1
    else:
        num = np.random.poisson(lam=1)
    dependents.append(num)

# 거주 형태
residence_status = np.random.choice(["Owner", "Renter", "Living with Parents"], size=size, p=[0.2, 0.6, 0.2])

# 이전 대출 이력
loan_history = np.random.choice(["No Previous Loan", "Paid in Full", "Defaulted"], size=size, p=[0.5, 0.4, 0.1])

# 기존 대출 잔고
existing_loan_balance = []
for history in loan_history:
    if history == "No Previous Loan":
        existing_loan_balance.append(0)
    elif history == "Paid in Full":
        balance = np.random.choice([0, np.random.lognormal(mean=np.log(1000), sigma=0.5)], p=[0.8, 0.2])
        existing_loan_balance.append(balance)
    elif history == "Defaulted":
        balance = np.random.lognormal(mean=np.log(15000), sigma=1.5)
        existing_loan_balance.append(balance)

# 은행 계좌 잔액
bank_balances = np.random.lognormal(mean=np.log(size), sigma=1, size=size).astype(int)

# 학력 수준
education_levels = np.random.choice(["High School", "Bachelor", "Master or above"], size=size, p=[0.4, 0.4, 0.2])

# 신용 점수
init_credit_scores = np.random.normal(650, 100, size=size).clip(300, 850).astype(int)

# 영향 요소 계산
impact_from_existing_loan = -0.1 * (np.array(existing_loan_balance) / (np.max(existing_loan_balance) + 1e-6))
impact_from_dti = -0.2 * (dti_ratios / np.max(dti_ratios))
impact_from_loan_history = np.where(np.array(loan_history) == "Defaulted", -100, np.where(np.array(loan_history) == "Paid in Full", 50, 0))
impact_from_bank_balance = 0.1 * (bank_balances / np.max(bank_balances))
impact_from_income = 0.1 * (incomes / np.max(incomes))
impact_from_employment_status = np.where(employment_status == "Employed", 50, 
                                         np.where(employment_status == "Self-employed", 20, -50))
impact_from_residence_status = np.where(residence_status == "Owner", 50, 
                                        np.where(residence_status == "Renter", -20, -50))

credit_scores = (init_credit_scores +
                 impact_from_existing_loan +
                 impact_from_dti +
                 impact_from_loan_history +
                 impact_from_bank_balance +
                 impact_from_income +
                 impact_from_employment_status +
                 impact_from_residence_status)
credit_scores = np.clip(credit_scores, 300, 800).astype(int)

# 최대 대출 한도
max_loan_limit = (credit_scores / 800) * (incomes / 2)
max_loan_limit = (max_loan_limit - np.array(existing_loan_balance)).astype(int)
max_loan_limit = np.maximum(max_loan_limit, 0)

# 대출 승인 가능성
approval_probabilities = (
    (credit_scores > 680).astype(int) * 0.25 +                                          # 신용 점수 영향
    (incomes > 60000).astype(int) * 0.2 +                                               # 소득 영향
    (employment_status == "Employed").astype(int) * 0.15 +                             	# 고용 상태 영향
    (dti_ratios < 0.33).astype(int) * 0.1 +                                             # DTI 비율 영향
    (residence_status == "Owner").astype(int) * 0.1 +                                   # 거주 형태 영향
    (np.array(existing_loan_balance) < np.array(max_loan_limit)).astype(int) * 0.1 +    # 기존 대출 잔고 영향
    (education_levels == "Bachelor").astype(int) * 0.05 +                               # 학력 수준 영향 (학사 학위)
    (education_levels == "Master or above").astype(int) * 0.1 +                         # 학력 수준 영향 (석사 이상)
    (loan_amounts <= max_loan_limit * 0.9).astype(int) * 0.05                           # 대출 한도 영향
)

# 대출 승인 여부
loan_approval_status = (approval_probabilities > 0.5).astype(int)


data = pd.DataFrame({
    "Credit Score": credit_scores,
    "Income": incomes,
    "Employment Status": employment_status,
    "Debt-to-Income Ratio": dti_ratios,
    "Loan Amount": loan_amounts,
    "Loan Purpose": loan_purposes,
    "Marital Status": marital_status,
    "Number of Dependents": dependents,
    "Residence Status": residence_status,
    "Previous Loan History": loan_history,
    "Bank Account Balance": bank_balances,
    "Education Level": education_levels,
    "Existing Loan Balance": existing_loan_balance,
    "Maximum Loan Limit": max_loan_limit,
    "Loan Approval Status": loan_approval_status
})

data.to_csv(file_name, header=True, index_label="Id")
