import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1, 1], [2, 1], [3, 1]])

print(np.repeat(a[:, None], 2, axis=1)*b)


'''
df = pd.read_csv("data.csv")
iterations = df["iterations"]
slopes = df["slope"]
intercepts = df["intercept"]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(iterations, slopes, color="red", s=1)
ax1.set_title("Slope vs Iteration")
ax2.scatter(iterations, intercepts, color="blue", s=1)
ax2.set_title("Intercept vs Iteration")
plt.show()
'''