import gurobipy as gp

# Create a new model
model = gp.Model()

# Define decision variables
x = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="x")
y = model.addVar(lb=0, ub=50, vtype=gp.GRB.CONTINUOUS, name="y")
z = model.addVar(lb=0, ub=50, vtype=gp.GRB.CONTINUOUS, name="z")
b = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="b")

# Set objective function
P = 5*x + 985*y - 475*z - 0.9*b
model.setObjective(P, gp.GRB.MAXIMIZE)

# Add constraints
budget_constraint = model.addConstr(20*x + 1000*y - 15*z + 0.9*b <= 20000, "budget")
min_profit_constraint_1 = model.addConstr(5*x + 985*y - 475*z - 0.9*b >= 2000, "min_profit_1")
min_profit_constraint_2 = model.addConstr(5*x + 595*y + 285*z - 0.9*b >= 2000, "min_profit_2")
min_profit_constraint_3 = model.addConstr(5*x + 175*y + 695*z - 0.9*b >= 2000, "min_profit_3")

# Additional constraints for minimum profit of £2,000 in any of the three scenarios
min_profit_constraint_1_mod = model.addConstr(5*x + 985*y - 475*z - 0.9*b >= 2000, "min_profit_1_mod")
min_profit_constraint_2_mod = model.addConstr(5*x + 595*y + 285*z - 0.9*b >= 2000, "min_profit_2_mod")
min_profit_constraint_3_mod = model.addConstr(5*x + 175*y + 695*z - 0.9*b >= 2000, "min_profit_3_mod")

# Solve the model
model.optimize()

# Print the optimal solution and objective value
for v in model.getVars():
    print(f"{v.varName} = {v.x}")
print(f"Optimal objective value: £ {model.objVal:.2f}")