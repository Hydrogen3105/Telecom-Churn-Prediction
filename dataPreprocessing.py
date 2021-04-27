import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("telecom_users.csv")
# print(df.info())

df_dev = df.iloc[:, 1:].copy()

le = LabelEncoder()
# Partner, Dependents, PhoneService, PaperlessBilling, Churn have Yes, No output
le.fit(['Yes', 'No'])
yes_no_classes = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for service in yes_no_classes:
    df_dev[service] = le.transform(df_dev[service])

# change gender Male = M, Female = F
df_dev['gender'] = ['M' if x == 'Male' else 'F' for x in df_dev['gender']]

# OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies has Yes, No, No Internet
le_online = LabelEncoder()
onlineServices = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

le_online.fit(df_dev['OnlineSecurity'])
for service in onlineServices:
    df_dev[service] = le_online.transform(df_dev[service])

# fill 0 in TotalCharges for tenure = 0 rows
df_dev.loc[df_dev['tenure'] == 0, 'TotalCharges'] = 0
