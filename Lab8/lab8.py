import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

data = pd.read_csv('Prices.csv')

price = data['Price']
processor_speed = data['Speed']
hard_drive_size = np.log(data['HardDrive'])

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + beta1 * processor_speed + beta2 * hard_drive_size

    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=price)

    trace = pm.sample(1000, tune=1000, cores=1)

#HDI
beta1_hdi = pm.hdi(trace.posterior['beta1'], hdi_prob=0.95)
beta2_hdi = pm.hdi(trace.posterior['beta2'], hdi_prob=0.95)

print(f"HDI pentru beta1: {beta1_hdi.beta1.data}")
print(f"HDI pentru beta1: {beta2_hdi.beta2.data}")

# 3. Frecventa procesorului si marimea hard diskului sunt predictori utili intrucat sunt rezultatele HDI sunt diferite de 0

with pm.Model() as model2:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + beta1 * 33 + beta2 * np.log(540)

    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=price)

    trace2 = pm.sample(5000, tune=1000, cores=1, return_inferencedata=True)

#HDI
hdi = pm.hdi(trace2.posterior['likelihood'], hdi_prob=0.90)
print(hdi)