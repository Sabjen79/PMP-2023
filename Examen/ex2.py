import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def posterior_grid(grid_points=10):
    grid = np.linspace(0.00001, 1, grid_points) # valoarea 0 imi rezulta NaN in stats.geom.pmf, deci folosesc 0.00001
    prior = np.repeat(1/grid_points, grid_points)
    likelihood = stats.geom.pmf(5, grid) # n =5 pentru a 5a aruncare
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

grid, posterior = posterior_grid(10000)
plt.plot(grid, posterior, 'o-')
plt.xlabel('θ')

print(f'Valoarea lui θ care maximizeaza probabilitatea a posteriori este θ = {grid[posterior.argmax()]}')

plt.show()