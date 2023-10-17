from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# C = cutremur ( C=1 cand e cutremur, 0 in caz contrar )
# I = incendiu ( I=1 cand e incendiu, 0 in caz contrar )
# A = alarma ( A=1 cand se declanseaza alarma, 0 in caz contrar )

model = BayesianNetwork([('C', 'I'), ('C', 'A'), ('I', 'A')])
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])


cpd_i = TabularCPD(variable='I', variable_card=2, 
                   values=[[0.99, 0.97], 
                           [0.01, 0.03]],
                   evidence=['C'],
                   evidence_card=[2])

cpd_a = TabularCPD(variable='A', variable_card=2,
                   values=[[0.9999, 0.98, 0.05, 0.02],
                           [0.0001, 0.02, 0.95, 0.98]],
                    evidence=['C', 'I'],
                    evidence_card=[2, 2])

model.add_cpds(cpd_c, cpd_i, cpd_a)
assert model.check_model()

print("Stiind ca alarma a fost declansata ( A = 1 ), probabilitatea ca un cutremur sa se fi intamplat ( C = 1 ) este:")
print( VariableElimination(model).query(variables=['C'], evidence={'A': 1}) )

print("Stiind ca alarma nu a fost declansata ( A = 0 ), probabilitatea ca un incendiu sa se fi intamplat ( I = 1 ) este:")
print( VariableElimination(model).query(variables=['I'], evidence={'A': 0}) )

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()
