import gurobipy as gp
from gurobipy import GRB

# Create a new model
m = gp.Model("portfolio_optimization")

# Define decision variables
B = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="B")
S = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="S")
C = m.addVar(lb=-50, ub=50, vtype=GRB.CONTINUOUS, name="C")

# Set objective function
m.setObjective(10*B + 4*S, GRB.MAXIMIZE)

# Add constraints
m.addConstr(90*B + 20*S + 1000*C <= 20000, name="budget")
m.addConstr(C >= 0, name="call_option_buy_limit")
m.addConstr(C <= 50, name="call_option_sell_limit")

# Define three additional variables
P1 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="P1")
P2 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="P2")
P3 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="P3")

# Set new objective function
m.setObjective((1/3)*P1 + (1/3)*P2 + (1/3)*P3, GRB.MAXIMIZE)

# Add additional constraints
m.addConstr(10*B + 20*S + 1500*C == P1, name="profit_scenario_1")
m.addConstr(10*B - 500*C == P2, name="profit_scenario_2")
m.addConstr(10*B - 8*S - 1000*C == P3, name="profit_scenario_3")
m.addConstr(P1 >= 2000, name="minimum_profit_scenario_1")
m.addConstr(P2 >= 2000, name="minimum_profit_scenario_2")
m.addConstr(P3 >= 2000, name="minimum_profit_scenario_3")

# Optimize model
m.optimize()

# Print results
print("Optimal solution:")
print("B = ", B.x)
print("S = ", S.x)
print("C = ", C.x)
print("Expected profit = £", m.objVal)

# Define new objective function
Z = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z")

# Set new objective function
m.setObjective(Z, GRB.MAXIMIZE)

# Add new constraints
m.addConstr(P1 >= Z, name="riskless_profit_scenario_1")
m.addConstr(P2 >= Z, name="riskless_profit_scenario_2")
m.addConstr(P3 >= Z, name="riskless_profit_scenario_3")

# Optimize model
m.optimize()

# Print results
print("Portfolio with maximum riskless profit:")
print("B = ", B.x)
print("S = ", S.x)
print("C = ", C.x)
print("Expected profit = £", m.objVal)
