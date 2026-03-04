import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import data from linear_regression.py
df = pd.read_csv("data.csv")
iterations = df["iterations"]
slopes = df["slope"]
intercepts = df["intercept"]

# Plot slope/intercept vs iteration
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(iterations, slopes, color="red", s=1)
ax1.set_title("Slope vs Iteration")
ax2.scatter(iterations, intercepts, color="blue", s=1)
ax2.set_title("Intercept vs Iteration")
plt.show()
