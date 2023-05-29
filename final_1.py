import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

def rosenbrock(x,y):
    a=1
    b=100
    return (a-x)**2 + b * (y - x**2 )**2

def PSO_opti(num_particles, num_generation):
    #defining the value of constants
    w = 0.75 #inertia weight
    c1 = 1.5 #cognitive weight
    c2 = 1.5 #social weight
    r1 = np.random.rand(num_particles,2)
    r2 = np.random.rand(num_particles,2)
    search_space = [-5,5]
    l=[]

    #initialise particals
    particles = np.random.uniform(search_space[0],search_space[1], size = (num_particles,2))
    velocities = np.zeros((num_particles,2))
    pBest_position = particles.copy()
    pBest_value = np.full(num_particles, np.inf)
    gBest_position = None
    gBest_value = np.inf
    print("\n\\\\\\\\\Iteration Start from here")

    # PSO iteration
    for iter in range(num_generation):
        # Evaluate the fitness of all the particles
        values = rosenbrock(particles[:,0],particles[:,1])

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
        velocities = (w * velocities) + (c1 * r1 * (pBest_position - particles)) + (c2 * r2 * (gBest_position - particles))
        particles += velocities

        # Keeping particles in bound
        particles = np.clip(particles,search_space[0],search_space[1])

        # Iteration information
        print(f"Iteration {iter + 1}: Best Value = {gBest_value:.4f}")
        l.append(gBest_value)

    print(l)
    data = np.array(l)
    plt.plot(data)
    plt.show()

    # Return the best solution found
    return gBest_position, gBest_value


bestPosition, bestValue = PSO_opti(10,100)
print("\nOptimization result")
print(f"Best Value : {bestValue}")
print(f"Position : {bestPosition}")
