import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Admission.csv')
print(df.head())

data = {
    'GRE': np.array(df['GRE']),
    'GPA': np.array(df['GPA']),
    'Admission': np.array(df['Admission'])
}

# Ex1
with pm.Model() as model:
    trace = pm.sample(1000, tune=1000)

mean_betas = [np.mean(trace['beta0']), np.mean(trace['beta1']), np.mean(trace['beta2'])]
decision = -mean_betas[0] / mean_betas[2] - (mean_betas[1] / mean_betas[2]) * data['GRE']

hdi = pm.stats.hpd(trace, hdi_prob=0.94)

# Ex2
plt.scatter(data['GRE'], data['GPA'], c=data['Admission'], cmap='viridis')
plt.plot(data['GRE'], decision, label='Decision Boundary', color='red')
plt.fill_between(data['GRE'], hdi[:, 1] / mean_betas[2] + hdi[:, 0] / mean_betas[2], alpha=0.3, color='blue', label='94% HDI')
plt.xlabel('GRE')
plt.ylabel('GPA')
plt.legend()
plt.show()

# Ex 3.
new_student_data_1 = {'GRE': 550, 'GPA': 3.5}
new_pi_1 = pm.math.sigmoid(mean_betas[0] + mean_betas[1] * new_student_data_1['GRE'] + mean_betas[2] * new_student_data_1['GPA'])
new_hdi_1 = pm.stats.hpd(new_pi_1)

# Ex 4
new_student_data_2 = {'GRE': 500, 'GPA': 3.2}
new_pi_2 = pm.math.sigmoid(mean_betas[0] + mean_betas[1] * new_student_data_2['GRE'] + mean_betas[2] * new_student_data_2['GPA'])
new_hdi_2 = pm.stats.hpd(new_pi_2)

print(f"Interval HDI pentru studentul 550 și GPA 3.5: {new_hdi_1}")
print(f"Interval HDI pentru studentul 500 și GPA 3.2: {new_hdi_2}")