import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom
from scipy import stats

# Ex 1
    
def posterior_grid(grid_points=50, heads=6, tails=9, prior=None):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)

    if prior is None:
        prior = np.repeat(1/grid_points, grid_points)  #uniform prior

    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h

prior = np.repeat(1/points, points)
grid, posterior = posterior_grid(points, h, t, prior)

# A priori - Mai mic decat 0.5
prior1 = (grid <= 0.5).astype(int)
grid1, posterior1 = posterior_grid(points, h, t, prior1)

# A priori - Distanta fata de 0.5
prior2 = np.abs(grid - 0.5)
grid2, posterior2 = posterior_grid(points, h, t, prior2)

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(grid, posterior, 'o-')
plt.title('Uniform Prior')
plt.xlabel('theta')
plt.ylabel('Posterior')

plt.subplot(132)
plt.plot(grid1, posterior1, 'o-')
plt.title('Prior <= 0.5')
plt.xlabel('theta')
plt.ylabel('Posterior')

plt.subplot(133)
plt.plot(grid2, posterior2, 'o-')
plt.title('Prior distance from 0.5')
plt.xlabel('theta')
plt.ylabel('Posterior')

plt.tight_layout()
plt.show()

# Ex 2

def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return pi, error


num_trials = 100
N_values = [100, 1000, 10000]

pi_estimates = []
errors = []

for N in N_values:
    pi_values = []
    error_values = []
    for _ in range(num_trials):
        pi, error = estimate_pi(N)
        pi_values.append(pi)
        error_values.append(error)

    mean_error = np.mean(error_values)
    std_error = np.std(error_values)

    pi_estimates.append(pi_values)
    errors.append((mean_error, std_error))


plt.figure(figsize=(10, 6))
for i, N in enumerate(N_values):
    plt.errorbar(N, errors[i][0], yerr=errors[i][1], fmt='o', label=f'N = {N}')

plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Error (%)')
plt.title('Estimation of pi')
plt.legend()
plt.show()