from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
from random import random

# F - primul jucator ( F = 0, incepe J0. F = 1, incepe J1)
# R1 - Runda 1 ( R1 = 0, jucatorul nu da stema, R2 = 1 jucatorul da stema)
# R2 - Runda 2 ( R2 = 0, jucatorul da 0 steme, R2 = 1 jucatorul da o stema, R2 = 2 Jucatorul da 2 steme )

j0_castig = 0 #nr de castiguri pentru J0
j1_castig = 0 #...                    J1
samples = 10000
for i in range(samples):
    f = 0 if random() < 1/2 else 1

    if f == 0: #J0 in runda1
        r1 = 0 if random() < 1/3 else 1 

        r2_r = random()
        if r1 == 0:
            r2 = 0 if r2_r < 1/2 else 1
        else:
            r2 = 0 if r2_r < 1/4 else 1 if r2_r < 2/4 else 2
        
    if f == 1: #J1 in runda 1
        r1 = 0 if random() < 1/2 else 1

        r2_r = random()
        if r1 == 0:
            r2 = 0 if r2_r < 1/3 else 1
        else:
            r2 = 0 if r2_r < 1/9 else 1 if r2_r < 4/9 else 2

    if f == 0: #J0 in runda 1
        j0_castig += 1 if r1 > r2 else 0
        j1_castig += 1 if r2 > r1 else 0

    if f == 1: #J1 in runda 1
        j0_castig += 1 if r2 > r1 else 0
        j1_castig += 1 if r1 > r2 else 0

print(f'Sanse jucator 1: {j0_castig/samples}\nSanse Jucator 2: {j1_castig/samples}\nSanse Egalitate: {(samples-j0_castig-j1_castig)/samples}\n')

model = BayesianNetwork([('F', 'R1'), ('F', 'R2'), ('R1', 'R2')])

cpd_f = TabularCPD(variable='F', variable_card=2, values=[[0.5], [0.5]]) #Sanse egale
cpd_r1 = TabularCPD(variable='R1', variable_card=2, 
                    values=[[1/3, 1/2], # Jucatorul J0 are sanse mai mari sa dea stema
                            [2/3, 1/2]],
                    evidence=['F'],
                    evidence_card=[2])
cpd_r2 = TabularCPD(variable='R2', variable_card=3,
                   values=[[1/2, 1/3, 1/4, 1/9], 
                           [1/2, 2/3, 2/4, 4/9],
                           [  0,   0, 1/4, 4/9]], #Daca in prima runda s-au dat 0 steme, e imposibil ca in a doua sa fie 2 steme
                    evidence=['F', 'R1'],
                    evidence_card=[2, 2])

model.add_cpds(cpd_f, cpd_r1, cpd_r2)
model.check_model()

infer = VariableElimination(model)
primul_jucator = infer.query(variables=['F'], evidence={'R2': 1}) #R2 - 1, s-a aruncat doar o stema
print('[F(x) = p] probabilitatea ca jucatorul x sa fi inceput stiind ca in r2 s-a dat o singura stema:\n', primul_jucator)