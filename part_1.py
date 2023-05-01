# Import the necessary libraries
import numpy as np
from scipy.optimize import minimize

# Define the decision variables
x = np.array([x_stock, x_bond, x_option])

# Define the objective function
def objective(x):
  # Calculate the profit from each asset
  profit_stock = x_stock * (40 - 20)
  profit_bond = x_bond * 0.05
  profit_option = x_option * (40 - 15) - 1000

  # Calculate the expected profit
  return np.average([profit_stock, profit_bond, profit_option])

# Define the constraints
constraints = [
  # The investor has £20,000 to invest
  x_stock + x_bond + x_option <= 20000,
  # The number of call options that the investor buys or sells is at most 50
  x_option <= 50,
  # The investor wants a profit of at least £2,000 in any of the three scenarios
  profit_stock >= 2000,
  profit_bond >= 2000,
  profit_option >= 2000
]

# Solve the linear program
solution = minimize(objective, x, constraints=constraints)

# Print the solution
print(solution)