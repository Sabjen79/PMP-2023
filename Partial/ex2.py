import pymc as pm
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

mu = 10
sigma = 2

timpi_asteptare = np.random.normal(mu, sigma, 100)

plt.hist(timpi_asteptare, bins=20, density=True, alpha=0.7)
plt.xlabel('Timp mediu de asteptare')
plt.ylabel('Frecventa')
plt.show()


with pm.Model() as model:
    mu_priori = pm.Normal('mu', mu=mu, sigma=5)
    sigma_priori = pm.HalfNormal('sigma', sigma=5)
    
    verosimilitate = pm.Normal('verosimilitate', mu=mu_priori, sigma=sigma_priori, observed=timpi_asteptare)
    
    trace = pm.sample(1000, step=pm.Metropolis(), tune=500, chains=1)

pm.plot_posterior(trace)
plt.show()

mu_posterior = np.mean(trace.posterior.mu)
print(f'Estimarea a posteriori pentru mu:\n {mu_posterior}') #Da aproximativ 10, rezultat corespunzator cu mu definit la inceput