import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

x_cubic = np.linspace(-1.5, 1.5, 500)
y_cubic = x_cubic**3 + np.random.normal(0, 1, size=500)

order_cubic = 3
x_cubic_p = np.vstack([x_cubic**i for i in range(1, order_cubic + 1)])
x_cubic_s = (x_cubic_p - x_cubic_p.mean(axis=1, keepdims=True)) / x_cubic_p.std(axis=1, keepdims=True)
y_cubic_s = (y_cubic - y_cubic.mean()) / y_cubic.std()

# Model cubic
with pm.Model() as model_cubic:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1, shape=order_cubic)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_cubic_s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_cubic_s)
    idata_cubic = pm.sample(2000, return_inferencedata=True)

# Model liniar
with pm.Model() as model_linear:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1, shape=1)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_cubic_s[0])
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_cubic_s)
    idata_linear = pm.sample(2000, return_inferencedata=True)

# Model patratic
with pm.Model() as model_quadratic:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1, shape=2)
    eps = pm.HalfNormal('eps', 5)
    mu = alpha + pm.math.dot(beta, x_cubic_s[:2])
    y_pred = pm.Normal('y_pred', mu=mu, sigma=eps, observed=y_cubic_s)
    idata_quadratic = pm.sample(2000, return_inferencedata=True)

plt.figure(figsize=(12, 8))

# WAIC
waic_cubic = az.waic(idata_cubic)
waic_linear = az.waic(idata_linear)
waic_quadratic = az.waic(idata_quadratic)
plt.subplot(2, 1, 1)
az.plot_waic([waic_cubic, waic_linear, waic_quadratic], labels=['Cubic', 'Linear', 'Quadratic'])
plt.title('WAIC')

# LOO
loo_cubic = az.loo(idata_cubic)
loo_linear = az.loo(idata_linear)
loo_quadratic = az.loo(idata_quadratic)
plt.subplot(2, 1, 2)
az.plot_loo([loo_cubic, loo_linear, loo_quadratic], labels=['Cubic', 'Linear', 'Quadratic'])
plt.title('LOO')

plt.tight_layout()
plt.show()