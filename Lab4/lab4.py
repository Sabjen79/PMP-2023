import numpy as np
import pymc as pm
import arviz
import matplotlib.pyplot as plt

clients_lambda = 20
time_mean = 2
time_deviation = 0.5

with pm.Model() as model:
    clients = pm.Poisson('C', mu=clients_lambda)
    time = pm.Normal('T', mu=time_mean, sigma=time_deviation)

with model:
    trace = pm.sample(1000, cores=1)

print(trace)