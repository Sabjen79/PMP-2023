import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats

latency = stats.expon(0, 1/4)
time1 = stats.gamma(4, 0, 1/3)
time2 = stats.gamma(4, 0, 1/2)
time3 = stats.gamma(5, 0, 1/2)
time4 = stats.gamma(5, 0, 1/3)

result = []

for _ in range(10000):
    server = np.random.choice([1, 2, 3, 4], p=[.25, .25, .3, .2])

    if server == 1:
        time = time1.rvs()
    elif server == 2:
        time = time2.rvs()
    elif server == 3:
        time = time3.rvs()
    else:
        time = time4.rvs()

    time = time + latency.rvs()
    result.append(time)

probability = np.mean(np.array(result) > 3)

print("Probabilitate > 3:", probability)

az.plot_posterior({'x':result})
plt.show() 
