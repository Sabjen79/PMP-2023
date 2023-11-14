import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

df = pd.read_csv('auto-mpg.csv')

columns = df[['mpg', 'horsepower']]

columns['mpg'] = pd.to_numeric(columns['mpg'], errors='coerce')
columns = columns.dropna(subset=['mpg'])

columns['horsepower'] = pd.to_numeric(columns['horsepower'], errors='coerce')
columns = columns.dropna(subset=['horsepower'])

mpg_data = columns['mpg'].values
hp_data = columns['horsepower'].values


with pm.Model() as model:
    alpha = pm.Normal('alpha', mu = 0)
    beta = pm.Normal('beta', mu = 0)

    mu = alpha + beta * hp_data
    prob = pm.Normal('mpg', mu=mu, observed=mpg_data)

    trace = pm.sample(100)

az.plot_posterior(trace)
plt.show()