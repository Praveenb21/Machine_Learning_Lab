import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # Fixed import

# Input data points
x = [[40, 20], [50, 50], [60, 90], [10, 25], [70, 70], [60, 10], [25, 80]]
y = ["red", "blue", "blue", "red", "blue", "red", "blue"]

# Query point
q = [[20, 35]]  # Should be 2D array

# Create and train the k-nearest neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x, y)

# Make prediction
predictions = knn.predict(q)
print("Predicted class for", q[0], "is", predictions[0])

# Plotting the data points
for (x1, y1), label in zip(x, y):
    color = "red" if label == "red" else "blue"
    plt.scatter(x1, y1, c=color)

# Plotting the query point
plt.scatter(q[0][0], q[0][1], c='green', marker='x', s=100, label="Query Point")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("K-Nearest Neighbors Classification")
plt.show()