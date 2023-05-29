import numpy as np

def rosenbrock(x, y):
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

def pso_optimization(num_particles, num_iterations):
    # PSO parameters
    inertia_weight = 0.729  # inertia weight
    cognitive_weight = 1.494  # cognitive weight
    social_weight = 1.494  # social weight

    search_space = (-5, 5)  # search space for both x and y coordinates

    # Initialize particles
    particles = np.random.uniform(search_space[0], search_space[1], size=(num_particles, 2))
    velocities = np.zeros((num_particles, 2))
    personal_best_positions = particles.copy()
    personal_best_values = np.full(num_particles, np.inf)
    global_best_position = None
    global_best_value = np.inf

    # PSO iterations
    for iteration in range(num_iterations):
        # Evaluate objective function for each particle
        values = rosenbrock(particles[:, 0], particles[:, 1])

        # Update personal best positions
        update_indices = values < personal_best_values
        personal_best_positions[update_indices] = particles[update_indices]
        personal_best_values[update_indices] = values[update_indices]

        # Update global best position
        best_particle_index = np.argmin(personal_best_values)
        if personal_best_values[best_particle_index] < global_best_value:
            global_best_position = personal_best_positions[best_particle_index]
            global_best_value = personal_best_values[best_particle_index]

        # Update velocities and positions
        r1 = np.random.rand(num_particles, 2)
        r2 = np.random.rand(num_particles, 2)
        velocities = (inertia_weight * velocities +
                      cognitive_weight * r1 * (personal_best_positions - particles) +
                      social_weight * r2 * (global_best_position - particles))
        particles += velocities

        # Clamp particles within the search space
        particles = np.clip(particles, search_space[0], search_space[1])

        # Print iteration information
        print(f"Iteration {iteration + 1}: Best Value = {global_best_value:.4f}")

    # Return the best solution found
    return global_best_position, global_best_value

# Example usage
best_position, best_value = pso_optimization(num_particles=30, num_iterations=100)
print("\nOptimization Result:")
print(f"Best Position: {best_position}")
print(f"Best Value: {best_value:.4f}")
