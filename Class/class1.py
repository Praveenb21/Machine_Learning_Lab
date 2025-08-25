import numpy as np
#arr = np.array([10,20,30,40])
#print("Summation", np.sum(arr))


#arr_1d = np.array([1,2,3,4,5,6])
#print("1D Array:", arr_1d)


# arr_2d = np.array([1, 2, 3, 4, 5, 6]).reshape(2,3)
# print("2D Array:\n", arr_2d)


# arr_3d = np.arange(1, 10).reshape(3, 3)
# print("3D Array:\n", arr_3d.T) 

# data = np.array([10, 20, 30, 40, 50])
# mean = np.sum(data)/len(data)
# variance = np.var(data)
# standard_deviation = np.std(data)
# print("Data:", data)
# print("Mean:", mean)   
# print("Variance:", variance)
# print("Standard Deviation:", standard_deviation) 


import matplotlib.pyplot as plt
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 30, 40, 50]
# plt.scatter(x, y, color='red')
# plt.xlabel('Study Hours')
# plt.ylabel('Scores')
# plt.title('Study Hours vs Scores')
# plt.grid("false")
# plt.show()


# from sklearn.linear_model import LinearRegression
# x = [[1], [2], [3], [4], [5], [6]]
# y = [10, 20, 30, 40, 50, 60]
# model = LinearRegression()
# model.fit(x, y)
# print(model.predict([[7]]))      # Predicting for a new value


import pandas as pd
# data = {'Name': ['Alice', 'Bob', 'Charlie'],  
#         'Age': [25, 30, 35],
#         'City': ['New York', 'Los Angeles', 'Chicago']  }
# pdf = pd.DataFrame(data)
# print(pdf)
# pdf.to_csv('data.csv', index=False)
# print("Summary", pdf.describe())



# x = np.array([1, 2, 3, 4, 5])
# y = np.array([6, 7, 8, 9, 10])
# cup = np.random.choice([1, 2, 3, 4, 5])
# sup = np.random.choice([6, 7, 8, 9, 10])
# print(f"{cup} + {sup} = ?")
# Result = cup + sup
# user_input = int(input("Try to guess the result: "))
# if user_input == Result:
#     print("Correct!")
# else:
#     print("Incorrect! The correct answer is:", Result)