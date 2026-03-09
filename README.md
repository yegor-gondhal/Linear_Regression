# Linear Regression
## Purpose
The purpose of this project was to demonstrate how linear regression using gradient-descent
causes the slope and y-intercept of a best-fit line to converge to values that 
minimize the MSE (Mean Squared Error).

## Files
The bmw_global_sales_2018_2025.csv file is just some arbitrary data that one could swap out.

The closed_form_OLS.py returns the slope and y-intercept that minimizes the MSE for the 
best-fit line that goes through the data. It does this by using the standard formula for
the closed-form ordinary least squares. Additionally, it returns a graph of the data along with
the best-fit line.

The linear_regression.py iterates over the slope and y-intercept until the values converge. It does
this through the gradient-descent formula, which repeatedly subtracts the derivative of the MSE
multiplied by some learning-rate from the original slope and intercept. This file prints the progress
in the console while running, and after finishing it saves the data in data.csv to avoid recomputation,
prints the final slope and intercept along with the values from closed_form_OLS.py for comparison,
and finally plots two graphs for the relationship between number of iterations and the value of the
slope or intercept.

The plot.py file creates the same graph that is produced by linear_regression.py using data.csv in
order to avoid recomputation.

## Libraries
- Numpy
- Pandas
- Matplotlib

## Useful Info
- When running linear_regression.py with the original bmw data that I used, it took about 1.1 million iterations.
- If you want to run this project using other data, make sure to correctly pre-process it as marked by the file.
  The correct format should be a pandas data file with two columns, x-axis data in the left column and y-axis
  data on the right.
