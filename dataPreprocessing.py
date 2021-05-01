import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

df = pd.read_csv("telecom_users.csv")
# print(df.info())

df_dev = df.iloc[:, 2:].copy()

le = LabelEncoder()
# Partner, Dependents, PhoneService, PaperlessBilling, Churn have Yes (1) , No (0) output
le.fit(['Yes', 'No'])
yes_no_classes = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for service in yes_no_classes:
    df_dev[service] = le.transform(df_dev[service]).astype('object')

# change gender Male = M, Female = F
df_dev['gender'] = ['M' if x == 'Male' else 'F' for x in df_dev['gender']]

# OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies has Yes (2), No (0), No Internet (1)
le_online = LabelEncoder()
onlineServices = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

le_online.fit(df_dev['OnlineSecurity'])
for service in onlineServices:
    df_dev[service] = le_online.transform(df_dev[service]).astype('object')

# fill 0 in TotalCharges for tenure = 0 rows
df_dev.loc[df_dev['tenure'] == 0, 'TotalCharges'] = 0

# set up appropriate type for each attributes
df_dev['SeniorCitizen'] = df_dev['SeniorCitizen'].astype('object')
df_dev['TotalCharges'] = df_dev['TotalCharges'].astype('float64')

# df_dev_numeric = df_dev.loc[:, ['tenure', 'MonthlyCharges', 'TotalCharges','Churn']].copy()

# find pearson's correlation
corr = df_dev.corr()
# print(corr.columns, corr.index)

# visualization
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
# plt.show()
#
# for feature in df_dev.columns:
#     if feature not in ['tenure', 'MonthlyCharges', 'TotalCharges']:
#         sns.countplot(x=df_dev[feature], data=df_dev)
#         plt.show()

# 4, 17, 18