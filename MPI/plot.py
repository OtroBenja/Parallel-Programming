#Sequential run
import numpy as np
import matplotlib.pyplot as plt

dataField = np.loadtxt('Rho_field.csv',delimiter=',')
fig, ax = plt.subplots(1,figsize=(12,6),dpi=100)
plot1 = ax.matshow(dataField)
fig.colorbar(plot1)
ax.axis('off')
fig.tight_layout()
fig.savefig('Rho_field.png')

dataField = np.loadtxt('Phi_solution.csv',delimiter=',')
fig, ax = plt.subplots(1,figsize=(12,6),dpi=100)
plot1 = ax.matshow(dataField)
fig.colorbar(plot1)
ax.axis('off')
fig.tight_layout()
fig.savefig('Phi_solution.png')
