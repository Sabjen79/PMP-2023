import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

x_1 = np.linspace(-1.5, 1.5, 500)
y_1 = 2 * x_1**3 - 4 * x_1**2 + 3 * x_1 + np.random.normal(0, 1, size=500)

order = 5
x_1p = np.vstack([x_1**i for i in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

# a
with pm.Model() as model_p:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.scatter(x_1s[0], y_1s)
    az.plot_posterior(idata_p)
    plt.title('Model cu sd=10')

# b
with pm.Model() as model_p_1:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=order) #sd 100
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

    plt.subplot(3, 1, 2)
    plt.scatter(x_1s[0], y_1s)
    az.plot_posterior(idata_p)
    plt.title('Model cu sd=100')

with pm.Model() as model_p_2:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order) #sd array
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

    plt.subplot(3, 1, 2)
    plt.scatter(x_1s[0], y_1s)
    az.plot_posterior(idata_p)
    plt.title('Model cu sd=100')


plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()