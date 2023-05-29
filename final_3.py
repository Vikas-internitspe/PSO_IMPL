import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(100)

def objective_fun(x, path):
    base_data = np.array(pd.read_csv(path))[:,1:]
    y = x.reshape(len(base_data),len(base_data))
    for i in range(len(y)):
        for j in range(len(y)):
            return np.sum(y[i][j] * np.log(y[i][j]/base_data[i][j])-1)

def PSO_function(num_particles, num_generation, path_base_data):
    w = 0.75
    c1 = 1.5
    c2 = 1.5
    base_data = np.array(pd.read_csv(path_base_data))[:, 1:]
    base_data_sum = np.sum(base_data)
    print(base_data)
    print(base_data_sum)
    num_variables = len(base_data) ** 2
    r1 = np.random.rand(num_particles, num_variables)
    r2 = np.random.rand(num_particles, num_variables)
    search_space = [0.1,600]
    l=[]


    #initialising particles
    particles = np.random.uniform(search_space[0],search_space[1], size = (num_particles, num_variables))
    print("Initializing the particles ----------------------------------")
    #print(particles)
    velocities = np.zeros((num_particles, num_variables))
    print("initializing the velocities ---------------------------------")
    #print(velocities)
    pBest_position  = particles.copy()
    pBest_value = np.full(num_particles, np.inf)
    print("Initial pBest position --------------------------------------")
    #print(pBest_position)
    gBest_position = None
    gBest_value = np.inf

    print("\nInitialization Complete--------------- Iteration starts here---\n")

    #PSO iteration
    for iter in range(num_generation):

        #Evaluating the fitness of all the particles
        values = []
        for i in particles:
            z = objective_fun(np.array(i),path_base_data)
            values.append(z)
        values = np.array(values)

        # Update the pBest position and value
        update_indices = values < pBest_value
        pBest_position[update_indices] = particles[update_indices]
        pBest_value[update_indices] = values[update_indices]

        # Update the gBest position and value
        gBest_index = np.argmin(pBest_value)
        if pBest_value[gBest_index]<gBest_value:
            gBest_value = pBest_value[gBest_index]
            gBest_position = pBest_position[gBest_index]

        # Updating the value of position and velocities
        velocities = (w * velocities) + (c1 * r1 * (pBest_position - particles)) + (
                    c2 * r2 * (gBest_position - particles))
        particles += velocities

        # Keeping particles in bound
        particles = np.clip(particles,search_space[0],search_space[1])

        # Iteration information
        print(f"Iteration {iter + 1}: Best Value = {gBest_value:.4f}")
        l.append(gBest_value)

    #print(l)
    data = np.array(l)
    plt.plot(data)
    plt.show()

    result_sum = np.sum(gBest_position)
    sum_ratio = base_data_sum / result_sum
    gBest_position2 = np.rint(gBest_position*sum_ratio).reshape(len(base_data),len(base_data))
    print(sum_ratio)

    return gBest_position2, gBest_value

bestPosition, bestValue = PSO_function(2,10,"sample.csv")
print("\nOptimization result")
print(f"Best Value : {bestValue}")
print(f"Position : \n{bestPosition}")