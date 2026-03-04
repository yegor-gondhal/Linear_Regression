import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("bmw_global_sales_2018_2025.csv")
# Clean-Up Data
columns_indices_to_drop = [2, 3, 5, 6, 7, 8, 9, 10]
columns_to_drop = []
for i in columns_indices_to_drop:
    columns_to_drop.append(data.columns[i])
data = data.drop(columns_to_drop, axis=1) # Leave only the year, month, and number of units sold
# Get total # of cars sold by year and month
data = data.groupby(["Year", "Month"], as_index=False)["Units_Sold"].sum()
# Add index column to act as number of months after beginning of 2018
data = data.reset_index()
# Leave only months after beginning of 2018 and # of cars sold
data = data.drop(["Year", "Month"], axis=1)
matrix = data.to_numpy()
#--------Manipulate the Data Above----------------
#--------Keep Operations Below the Same-----------
x = matrix[:,0]
y = matrix[:,1]
# Add a ones column to get the slope along with the intercept
op_x = np.column_stack((np.ones(len(x)), x))
ans = np.matmul(np.transpose(op_x), op_x)
ans = np.linalg.inv(ans)
ans = np.matmul(ans, np.transpose(op_x))
ans = np.matmul(ans, y)


theta_0 = ans[0]
theta_1 = ans[1]
print("Intercept: ", theta_0)
print("Slope: ", theta_1)
y_pred = theta_0 + theta_1*x
mse = np.sum((y - y_pred)**2)/np.size(y)
print("MSE: ", mse)

plt.figure()
plt.scatter(x, y)
x_line = np.linspace(np.min(x), np.max(x), 100)
y_line = theta_0 + theta_1*x_line
plt.plot(x_line, y_line, color="red")
plt.title("BMW Car Sales per Month Since 2018")
plt.xlabel("Months after January 1st, 2018")
plt.ylabel("Number of Cars Sold")
plt.show()
