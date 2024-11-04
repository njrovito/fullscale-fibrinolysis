import matplotlib.pyplot as plt
import numpy as np
import sys

Q_0 = 650/60
alpha = 0.8
T = 0.3099996187977

t = 0 
dt = 0.001
Q = 0
Q_list = []
t_list = []


	
cardiac_cycle_period = 0.9
systolic_time = cardiac_cycle_period / 3

# Q1
'''
while t < 1.8:
    t += dt
    # Define time relative to cycle (tau is time within cycle)
    tau = round(t % cardiac_cycle_period, 3)
    # Solve for total inlet flowrate
    if tau < systolic_time:
        flowrate = Q_0 * np.sin(np.pi * tau / 0.3)
    else:
        flowrate = 0
    Q_list.append(flowrate)
    t_list.append(t)
'''

# Q2
while t < 1.8:
    t += dt
    # Define time relative to cycle (tau is time within cycle)
    tau = round(t % cardiac_cycle_period, 3)
    # Solve for total inlet flowrate
    if tau < systolic_time:
        flowrate = alpha * Q_0 * np.sin(np.pi * tau / T)
    else:
        flowrate = -alpha * Q_0 * np.sin(systolic_time * np.pi / T) / (cardiac_cycle_period - T) * (tau - T) + alpha * Q_0 * np.sin(systolic_time * np.pi / T)
    Q_list.append(flowrate)
    t_list.append(t)





x = np.array(t_list)
y = np.array(Q_list)

plt.rc('text', usetex=True)
plt.rc('axes', linewidth=1)
plt.rc('font', weight = 'bold')
plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'

plt.plot(x, y, color='black', linestyle = '--', linewidth=2)
plt.xlabel('$\\mathbf{t~[s]}$', fontsize=18, fontweight='bold')
plt.ylabel('$\\mathbf{Q_{in}~\\mathbf{\\left[\\frac{mL}{s}\\right]}}$', fontsize=18, fontweight='bold')
# plt.title('\\textbf{Total Inlet Flowrate}', fontsize=20)
plt.savefig('total_flowrate.png')
plt.show()