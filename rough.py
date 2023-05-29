from my_utils import PSO_function as pso


bestPosition, bestValue = pso(2,10,"sample.csv")
print("Optimization results: \n")
print(f"Best Value : {bestValue}")
print(f"Position : \n{bestPosition}")