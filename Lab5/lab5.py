import pymc as pm
import numpy as np
import csv

file = open("trafic.csv", "r")
data = [int(x[1]) for x in list(csv.reader(file, delimiter=','))[1:]]
file.close()

with pm.Model() as model:
    alpha = 1.0/np.mean(data)

    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)

    tau = pm.DiscreteUniform("tau", lower=0, upper=len(data)-1)

with model:
    idx = np.arange(len(data))
    lambda_ = pm.math.switch(idx < tau, lambda_2 + 10, lambda_1)
    lambda_ = pm.math.switch(idx > tau, lambda_3 - 10, lambda_1)

    observ = pm.Poisson("obs", lambda_, observed=data)

with model:
    trace = pm.sample(10000, cores=1, step=pm.Metropolis(), return_inferencedata=False)
    print(trace['lambda_1'])
    print(trace['lambda_2'])
    print(trace['lambda_3'])

