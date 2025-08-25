import pandas as pd
import numpy as np
data = {'age': [27, 28, 29, None, 30, 31, None, 32, 33, 40],
        'height': [160, 161, 162, None, 163, 164, 165, None, 166, 170],
        'weight': [54, 56, 58, 59, None, 60, 61, None, 62, 63],
        'city': ['Kolkata', 'UP', 'Kolkata', 'Bihar', np.nan, 'China', 'Kolkata', 'Kanpur', 'Mathura', np.nan],
        'marks': [80, 90, 88, 86, None, 88, 92, 93, None, 94]}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)
print("missing values:\n", df.isnull().sum())
df_drop = df.dropna()
print(df_drop)
df_drop.to_csv('user.csv')
df['age'].fillna(df['age'].mean(), inplace=True)
df['height'].fillna(df['height'].median(), inplace=True)
df['weight'].fillna(method='ffill', inplace=True)
df['city'].fillna(df['city'].mode()[0], inplace=True)
df['marks'].fillna(method='bfill', inplace=True)
print("DataFrame after handling missing values:\n", df)
print("missing values after handling:\n", df.isnull().sum())