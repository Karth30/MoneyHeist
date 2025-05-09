## Objective 
To build a machine learning model that can predict CIBIL scores based on customer features like income, loan history, credit inquiries, and more. This score helps financial institutions evaluate an individual's creditworthiness.

--- 

## Dataset
Source: External_Cibil_Dataset.zip
Description: Contains historical customer financial records including features such as:
Age, Income, Loan Amount, Credit Utilization, Number of Credit Inquiries, Existing Debt, Loan Repayment History
Target Variable: CIBIL Score (numerical)

---

## Notebook Overview
File: moneyheist.ipynb

Steps:
Data Preprocessing & Cleaning
Feature Engineering
Model Training ( Random Forest)
Evaluation (R² score = 0.8841, MAE, RMSE)
Exporting the final model as pkl file

---

## Web App (Streamlit)
A Streamlit interface allows users to input:
Monthly income, Credit history, Loan amount, etc.
And get a predicted CIBIL score instantly.

![image](https://github.com/user-attachments/assets/8f7d2343-40ed-4352-8a32-052d3e404400)
![image](https://github.com/user-attachments/assets/e931d25d-766f-44da-a3ee-350b1cf8923c)




