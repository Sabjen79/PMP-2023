import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

#Ex 1

means = [4, 0, -4]
std_devs = [2, 2.5, 3]
weights = [0.4, 0.3, 0.1]

num_samples = 500
data = np.concatenate([np.random.normal(mean, std_dev, int(weight * num_samples))
    for mean, std_dev, weight in zip(means, std_devs, weights)])

np.random.shuffle(data)

plt.hist(data, bins=30, density=True, alpha=0.5, color='b')
plt.title('3 distributii gausiene')
plt.xlabel('Valoare')
plt.ylabel('Densitate')
plt.show()

# Ex 2

final_data = []
for n in [2, 3, 4]:
    with pm.Model() as model:
        weights = pm.Dirichlet('weights', a=np.ones(n))
        means = pm.Normal('means', mu=np.linspace(data.min(), data.max(), n), sigma=10, shape=n)
        sigma = pm.HalfNormal('sigma', sigma=10)
        y_obs = pm.NormalMixture('y_obs', w=weights, mu=means, sigma=sigma, observed=data)
        trace = pm.sample(1000, return_inferencedata=True, random_seed=123, cores=1, idata_kwargs={'log_likelihood': True})  
        p = pm.sample_posterior_predictive(trace)
        final_data.append(trace)

# Ex 3 (Observam ca, in cazul nostru, primul model este cel mai bun)
        
loocv = az.compare({
    'model1': final_data[0], 
    'model2': final_data[1], 
    'model3': final_data[2]
}, method='stacking', ic='loo', scale='deviance')

print(loocv)

waic = az.compare({
    'model1': final_data[0], 
    'model2': final_data[1], 
    'model3': final_data[2]
}, method='stacking', ic='waic', scale='deviance')

print(waic)