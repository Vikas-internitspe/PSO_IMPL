import random
import math

# Define the objective function
def objective_function(x):
    return x**2 - 10*x + 25 + x*x*x*x

# Define the constraint function
def constraint_function(x):
    return x - 5

# Define the penalty function
def penalty_function(x):
    return math.exp(abs(x)) - 1

# Define the PSO parameters
swarm_size = 20
max_iterations = 100
inertia_weight = 0.8
c1 = 2.0
c2 = 2.0
velocity_limits = [-1, 1]
penalty_factor = 100

# Initialize the particles
particles = []
for i in range(swarm_size):
    position = random.uniform(0, 5)
    velocity = random.uniform(velocity_limits[0], velocity_limits[1])
    particles.append({'position': position, 'velocity': velocity, 'personal_best_position': position})

# Initialize the global best position
global_best_position = particles[0]['position']

# Run the PSO algorithm
for iteration in range(max_iterations):
    for particle in particles:
        # Evaluate the objective function and constraints
        constraint_value = constraint_function(particle['position'])
        if constraint_value <= 0:
            fitness = objective_function(particle['position'])
        else:
            # Apply penalization for constraint violation
            fitness = objective_function(particle['position']) + penalty_factor * penalty_function(constraint_value)

        # Update the personal best position
        if fitness < objective_function(particle['personal_best_position']):
            particle['personal_best_position'] = particle['position']

        # Update the global best position
        if objective_function(particle['personal_best_position']) < objective_function(global_best_position):
            global_best_position = particle['personal_best_position']

        # Update the velocity and position
        updated_velocity = (inertia_weight * particle['velocity'] +
                            c1 * random.random() * (particle['personal_best_position'] - particle['position']) +
                            c2 * random.random() * (global_best_position - particle['position']))
        updated_position = particle['position'] + updated_velocity

        # Apply velocity limits
        updated_velocity = max(velocity_limits[0], min(updated_velocity, velocity_limits[1]))

        # Update the particle's velocity and position
        particle['velocity'] = updated_velocity
        particle['position'] = updated_position

# Print the best solution found
print("Best solution:", global_best_position)
print("Objective value:", objective_function(global_best_position))
