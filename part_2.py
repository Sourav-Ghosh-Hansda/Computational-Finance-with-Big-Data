# Import the necessary libraries.
import numpy as np
from scipy.optimize import minimize

# Define the parameters of the problem.
num_shares = 100
strike_price = 15
option_price = 1000
risk_free_rate = 0.05
stock_price = 20

# Define the possible scenarios for the price of stock XYZ six months from today.
scenarios = [12, 20, 40]
probabilities = [0.33, 0.33, 0.33]

# Define the objective function.
def objective(x):
  # Calculate the expected profit of the portfolio.
  profit = 0
  for scenario, probability in zip(scenarios, probabilities):
    # Calculate the number of shares of stock XYZ to buy.
    shares = x[0]
    # Calculate the number of call options to buy.
    options = x[1]
    # Calculate the profit from the stock.
    profit += shares * (scenario - stock_price)
    # Calculate the profit from the options.
    profit += options * (scenario - strike_price)
    # Calculate the profit from the risk-free bond.
    profit += risk_free_rate * (x[2] * 100)
  return profit

# Define the constraints.
constraints = [
  # The number of shares of stock XYZ must be non-negative.
  x[0] >= 0,
  # The number of call options must be non-negative.
  x[1] >= 0,
  # The number of shares of stock XYZ plus the number of call options must be less than or equal to 200.
  x[0] + x[1] <= 200,
  # The amount invested in the stock must be less than or equal to the initial investment.
  x[0] * stock_price <= 20000,
  # The amount invested in the options must be less than or equal to the initial investment.
  x[1] * option_price <= 20000,
  # The amount invested in the risk-free bond must be less than or equal to the initial investment.
  x[2] <= 20000,
  # The number of call options must be less than or equal to 50.
  x[1] <= 50,
]

# Solve the linear program.
solution = minimize(objective, x0=[100, 0, 19000], constraints=constraints)

# Print the solution.
print(solution)