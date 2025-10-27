import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Dataset
data = {
    'Outlook': [
        'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast',
        'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast', 'Sunny'
    ],
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
             'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
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

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

print("\nAfter Encoding:\n", df)


X = df[['Outlook', 'Temp', 'Humidity', 'Windy']]
y = df['Play Golf']

model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=['Outlook', 'Temp', 'Humidity', 'Windy'],
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Play Golf Dataset")
plt.show()
fig,ax = plt.subplots(figsize=(12,8))
plot_tree(model, feature_names = X.columns, class_names=['Yes','No'], filled=True,ax = ax )
plt.show()