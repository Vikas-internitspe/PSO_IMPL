import numpy as np
from pyswarms.single import GlobalBestPSO
from pyswarm import pso

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

def con(x):
    return [np.sum(x)-2896.]


# Set the bounds for each dimension of the ndarray
lb,ub = (np.array([0.1] * num_variables), np.array([5] * num_variables))

xopt, fopt = pso(custom_objective_function, lb, ub, f_ieqcons=con)

print("xopt",xopt)
print("fopt",fopt)
print("sum_result", np.sum(xopt))