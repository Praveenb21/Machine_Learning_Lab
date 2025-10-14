import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt

data = {
    'Outlook': [
        'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast',
        'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast', 'Sunny'
    ],
    'Temp.': [
        'Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
        'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'
    ],
    'Humidity': [
        'High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
        'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'
    ],
    'Windy': [
        False, True, False, False, False, True, True,
        False, False, False, True, True, False, True
    ],
    'Play Golf': [
        'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
        'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display dataset
print(df)