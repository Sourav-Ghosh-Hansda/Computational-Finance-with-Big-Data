import gurobipy as gp
from gurobipy import GRB

# Create a new model
m = gp.Model()

# Define decision variables
B = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="B")
S = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="S")
C = m.addVar(lb=-50, ub=50, vtype=GRB.CONTINUOUS, name="C")

# Set objective function
m.setObjective(10*B + 4*S, sense=GRB.MAXIMIZE)

# Add budget constraint
m.addConstr(90*B + 20*S + 1000*C <= 20000, name="budget")

# Add limit on number of call options
m.addConstr(C >= -50, name="call_options_lower")
m.addConstr(C <= 50, name="call_options_upper")

# Solve the model
m.optimize()

# Print the optimal solution and objective value
print("Optimal Solution:")
for v in m.getVars():
    print(v.varName, v.x)
print("Expected Profit: £", m.objVal)

# Add the additional constraint
P1 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="P1")
P2 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="P2")
P3 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="P3")

m.addConstr(10*B + 20*S + 1500*C == P1, name="profit_scenario_1")
m.addConstr(10*B - 500*C == P2, name="profit_scenario_2")
m.addConstr(10*B - 8*S - 1000*C == P3, name="profit_scenario_3")
m.addConstr(P1 >= 2000, name="min_profit_scenario_1")
m.addConstr(P2 >= 2000, name="min_profit_scenario_2")
m.addConstr(P3 >= 2000, name="min_profit_scenario_3")

# Set new objective function
m.setObjective(1/3 * (P1 + P2 + P3), sense=GRB.MAXIMIZE)

# Solve the model with the additional constraint
m.optimize()

# Print the optimal solution and objective value
print("\nOptimal Solution with Additional Constraint:")
for v in m.getVars():
    print(v.varName, v.x)
print("Expected Profit: £", m.objVal)

# Add the riskless profit constraint
Z = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z")

m.addConstr(P1 >= Z, name="riskless_profit_scenario_1")
m.addConstr(P2 >= Z, name="riskless_profit_scenario_2")
m.addConstr(P3 >= Z, name="riskless_profit_scenario_3")

# Set new objective function
m.setObjective(Z, sense=GRB.MAXIMIZE)

# Solve the model with the riskless profit constraint
m.optimize()

# Print the optimal solution and objective value
print("\nOptimal Solution with Riskless Profit Constraint:")
for v in m.getVars():
    print(v.varName, v.x)
print("Riskless Profit: £", m.objVal)
