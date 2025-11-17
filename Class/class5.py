import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# Dataset
data = {'Age': ['Young', 'Young', 'Middle', 'Senior', 'Senior', 'Middle'],
        'Obesity': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
        'BP': ['High', 'Normal', 'High',  'Normal', 'Normal', 'High'],
        'Diabetes': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No']}
df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# Label Encoding
le_age, le_ob, le_bp, le_dia = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()

# Transform each column separately and build a proper DataFrame for X
age_encoded = le_age.fit_transform(df['Age'])
ob_encoded = le_ob.fit_transform(df['Obesity'])
bp_encoded = le_bp.fit_transform(df['BP'])

X = pd.DataFrame({
        'Age': age_encoded,
        'Obesity': ob_encoded,
        'BP': bp_encoded
})

# Y should be a 1-D array/Series of labels (not a malformed DataFrame constructor call)
Y = pd.Series(le_dia.fit_transform(df['Diabetes']), name='Diabetes')

print("\nAfter Encoding:\n", X)
print("\nEncoded targets:\n", Y)