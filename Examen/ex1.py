import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt

# a - DataFrame
Dataset = pd.read_csv('BostonHousing.csv')
rm = Dataset['rm'].values.astype(float)
crim = Dataset['crim'].values.astype(float)
indus = Dataset['indus'].values.astype(float)
medv = Dataset['medv'].values.astype(float)

# b - rm, crim, indus in raport cu medv

for data in [rm, crim, indus]:
    with pm.Model() as model_regression:        # b
        alfa = pm.Normal('alfa', mu=0, sigma=1000)
        beta = pm.Normal('beta', mu=0, sigma=1000)
        eps = pm.HalfCauchy('eps', 5000)
        niu = pm.Deterministic('niu', data * beta + alfa)
        medv_pred = pm.Normal('medv_pred', mu=niu, sigma=eps, observed=medv)
        idata = pm.sample(2000, return_inferencedata=True)

    az.plot_trace(idata, var_names=['alfa', 'beta', 'eps'])
    plt.show()

# c - 95% HDI
