# Import Gurobi module
import gurobipy as gp

# Create a new optimization model
m = gp.Model()

# Define decision variables
B = m.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="B")
S = m.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="S")
C = m.addVar(lb=-50, ub=50, vtype=gp.GRB.CONTINUOUS, name="C")

# Set objective function
m.setObjective(10*B + 4*S, gp.GRB.MAXIMIZE)

# Add budget constraint
m.addConstr(90*B + 20*S + 1000*C <= 20000, "budget")

# Set limit on the number of call options
m.addConstr(C >= -50, "call_buy_limit")
m.addConstr(C <= 50, "call_sell_limit")

# Solve the model
m.optimize()

# Print solution
print(f"Optimal solution: B = {B.x}, S = {S.x}, C = {C.x}")
print(f"Expected profit: {m.objVal}")

