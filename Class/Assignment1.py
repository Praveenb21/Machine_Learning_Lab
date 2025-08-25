# Question 1: Data Preprocessing for Health Monitoring Dataset

import pandas as pd
import numpy as np

health_data = pd.DataFrame({
    'Patient_ID': [101, 102, 103, 104, 105, 106, 107],
    'Age': [25, 30, 28, np.nan, 45, 50, 29],
    'Weight': [58, 75, np.nan, 55, 85, 90, np.nan],
    'BloodPressure': [120, np.nan, 130, 110, 140, 150, np.nan],
    'Cholesterol': [200, 210, 220, 190, np.nan, 250, 230],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female']
})


missing_counts = health_data.isnull().sum()
print("Missing values per column:\n", missing_counts)


health_data.fillna({
    'Age': health_data['Age'].mean(),
    'Weight': health_data['Weight'].median(),
    'BloodPressure': health_data['BloodPressure'].mean(),
    'Cholesterol': health_data['Cholesterol'].median()
}, inplace=True)
print("Data after handling missing values:\n", health_data)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
health_data['Gender_encoded'] = le.fit_transform(health_data['Gender'])
print("Label Encoding:\n", health_data[['Gender', 'Gender_encoded']])

from sklearn.preprocessing import MinMaxScaler, StandardScaler


scaler = MinMaxScaler()
health_data[['Age_scaled', 'Weight_scaled']] = scaler.fit_transform(health_data[['Age', 'Weight']])

std_scaler = StandardScaler()
health_data['Cholesterol_std'] = std_scaler.fit_transform(health_data[['Cholesterol']])


print("Cholesterol mean:", round(health_data['Cholesterol_std'].mean(), 2))
print("Cholesterol std:", round(health_data['Cholesterol_std'].std(), 2))

# Question 2: 