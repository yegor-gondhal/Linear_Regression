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
x = matrix[:,0]
y = matrix[:,1]
# Add a ones column to act as the coefficient on theta_0 (intercept)
x = np.column_stack((np.ones(len(x)), x))
thetas = np.array([1, 1], dtype=np.float64) # Initialize theta_0 and theta_1 as 1
eta = 1e-4 # Learning rate
m = np.size(y)
counter = 0
prev_intercept = 0
prev_slope = 0
# Initialize lists for data points
slope_list = []
intercept_list = []
iteration_list = []
while True:
    y_pred = thetas[0]*x[:, 0] + thetas[1]*x[:, 1] # Predicted y using current intercept and slope
    #thetas[0] = thetas[0] - eta*np.sum((y_pred - y)*x[:, 0])/m
    #thetas[1] = thetas[1] - eta * np.sum((y_pred - y) * x[:, 1]) / m
    thetas = thetas - eta * np.sum(np.repeat((y_pred - y)[:, None], 2, axis=-1)*x, axis=0) / m
    if counter % 500 == 0: # Add new data points every 500 iterations
        slope_list.append(thetas[1])
        intercept_list.append(thetas[0])
        iteration_list.append(counter)

    if counter % 50000 == 0: # Record progress in the terminal
        print("Iteration: ", counter) # Thetas converges in about 1,100,000 iterations
        print("Intercept: ", thetas[0])
        print("Slope: ", thetas[1])
        mse = np.sum((y - y_pred)**2)/m
        print("MSE: ", mse)
        print("\n")
        if thetas[0] == prev_intercept and thetas[1] == prev_slope:
            break # Exit the loop once the intercept and slope converge
        # Re-assign the previous values if loop is not exited
        prev_intercept = thetas[0]
        prev_slope = thetas[1]
    counter += 1


print("Converged")
print("Final Slope", thetas[1])
print("Slope from OLS: 609.8525976668459", )
print("Final Intercept", thetas[0])
print("Intercept from OLS: 226401.22036082475")

# Convert python list to np array for easier calculation
iteration_list = np.array(iteration_list)
slope_list = np.array(slope_list)
intercept_list = np.array(intercept_list)

df = {
    "iterations": iteration_list,
    "slope": slope_list,
    "intercept": intercept_list,
}

df = pd.DataFrame(df)
df.to_csv("data.csv", index=False) # Export data

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(iteration_list, slope_list, color="red", s=1)
ax1.set_title("Slope vs Iteration")
ax2.scatter(iteration_list, intercept_list, color="blue", s=1)
ax2.set_title("Intercept vs Iteration")
plt.show()