import numpy as np
from pyswarms.single import GlobalBestPSO


# Define your custom objective function
def custom_objective_function(x):
    result = np.sum(x ** 2)
    return result


# Define your custom constraint function
def custom_constraint_function(x):
    # Apply constraints here
    constraint1 = np.sum(x) - 1  # Constraint 1: Sum of variables equals 1
    constraint2 = x[0] - x[1]  # Constraint 2: Variable 1 should be greater than Variable 2

    # Penalty function to handle violated constraints
    penalty = 0.0
    if constraint1 < 0:
        penalty += abs(constraint1)
    if constraint2 < 0:
        penalty += abs(constraint2)

    return penalty


# Set the bounds for each dimension of the ndarray
bounds = (np.array([0, 0]), np.array([1, 1]))

# Set the options and parameters for the PSO algorithm
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Initialize the PSO optimizer
optimizer = GlobalBestPSO(n_particles=10, dimensions=2, bounds=bounds, options=options)

# Set the custom constraint function
optimizer.set_constraints(custom_constraint_function)

# Run the optimization process
best_cost, best_position = optimizer.optimize(custom_objective_function, iters=100)

print("Best cost:", best_cost)
print("Best position:", best_position)
