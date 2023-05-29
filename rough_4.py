import numpy as np
from pyswarms.single import GlobalBestPSO

np.random.seed(100)

# Load the CSV file into a NumPy array
csv_file = 'sample.csv'
data = np.genfromtxt(csv_file, delimiter=',')[1:, 1:]
sum_base_data = np.sum(data)
print(sum_base_data)
base_data = data.ravel()
num_variables = len(base_data)

def custom_objective_function(x):
    global base_data
    objective_value = np.sum(x * (np.log(x / base_data) - 1))
    return objective_value

def custom_repair_function(x):
    # Normalize x to sum to 1
    x = x / np.sum(x) * 2896.4
    return x

# Set the bounds for each dimension of the ndarray
bounds = (np.array([0.1] * num_variables), np.array([500] * num_variables))

# Set the options and parameters for the PSO algorithm
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Initialize the PSO optimizer
optimizer = GlobalBestPSO(n_particles=10, dimensions=num_variables, bounds=bounds, options=options)

# Override the update position function to apply the repair function
def update_position_repair(particle, bounds):
    new_position = particle.position.copy()
    new_position = custom_repair_function(new_position)
    particle.position = new_position
    particle.position = np.clip(particle.position, bounds[0], bounds[1])

optimizer._update_position = update_position_repair

# Run the optimization process
best_cost, best_position = optimizer.optimize(custom_objective_function, iters=100)

# Apply repair function to the best position
best_position = custom_repair_function(best_position)
best_position = best_position / np.sum(best_position) * 2896.4 # Normalize again to ensure sum is exactly 1

sum_res_data = np.sum(best_position)
print("sum_result: ",sum_res_data)

# Converting the output to integer matrix format
#best_position2 = np.rint(best_position).reshape(len(data),len(data))

print("Best cost:", best_cost)
print("Best position: \n", best_position)