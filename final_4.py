import numpy as np
from pyswarms.single import GlobalBestPSO

np.random.seed(100)

# Load the CSV file into a NumPy array
csv_file = 'sample.csv'
data = np.genfromtxt(csv_file, delimiter=',')[1:, 1:]
sum_base_data = np.sum(data)
print("Sum_base: ",sum_base_data)
base_data = data.ravel()
num_variables = len(base_data)

# Define your custom objective function
def custom_objective_function(x):
    global base_data
    # Perform your calculations based on the elements of the ndarray
    result = np.sum(x * (np.log(x/base_data)-1))
    return result

# Set the bounds for each dimension of the ndarray
bounds = (np.array([0.1] * num_variables), np.array([5] * num_variables))

# Set the options and parameters for the PSO algorithm
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Initialize the PSO optimizer
optimizer = GlobalBestPSO(n_particles=10, dimensions=num_variables, bounds=bounds, options=options)

# Run the optimization process
best_cost, best_position = optimizer.optimize(custom_objective_function, iters=100)
sum_res_data = np.sum(best_position)
print("sum_result: ",sum_res_data)
# Converting the output to integer matrix format
best_position2 = np.rint(best_position).reshape(len(data),len(data))
best_position3 = np.rint(best_position*sum_base_data/sum_res_data).reshape(len(data),len(data))

print("Best cost:", best_cost)
print("Best position: \n", best_position2)
print("Best position: \n", best_position3)
