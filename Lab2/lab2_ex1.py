import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import arviz as az

time1 = stats.expon(scale=1/4)
time2 = stats.expon(scale=1/6)

result = []

for _ in range(10000):
    mecanic1 = np.random.rand() < 0.4

    if(mecanic1 == True):
        time = time1.rvs()
    else:
        time = time2.rvs()

    result.append(time)

print("Media lui X:", np.mean(result))
print("Deviatia standard a lui X:", np.std(result))

az.plot_posterior({'x':result})
plt.show() 