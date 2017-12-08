""" Neuron Simulator for CS 443 """
from neurons import LIF, Izhikevich, HodgkinHuxley
from neurons import TIME_STEPS, IZHIKEVICH_SPIKE, SPIKE_MULTIPLIER
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

CURRENTS = []
RESISTANCE = 2
CAPACITANCE = 2.5
REST_VOLTAGE = 10
SPIKE_VOLTAGE = 75


print('Problem 1')
plt.figure(1)
for x in range(10, 90, 20):
    CURRENTS.append(x)
for i in range(1, len(CURRENTS)+1):
    neuron = LIF(CAPACITANCE, RESISTANCE, SPIKE_VOLTAGE, REST_VOLTAGE)
    current = CURRENTS[i-1]
    plt.subplot(810+i)
    plt.title('Current: {}'.format(current))
    for i in range(TIME_STEPS-1):
        neuron.time_step(current, i)
    neuron.add_plot(plt)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, SPIKE_VOLTAGE*SPIKE_MULTIPLIER))
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.grid(True)
    plt.minorticks_on()
    plt.gca().set_yticks([REST_VOLTAGE, SPIKE_VOLTAGE])
    plt.gca().set_xticks([])
for i in range(1, len(CURRENTS)+1):
    neuron = LIF(CAPACITANCE, RESISTANCE, SPIKE_VOLTAGE, REST_VOLTAGE)
    current = CURRENTS[i-1]
    plt.subplot(810+i+4)
    plt.title('Current: {}'.format(current))
    for i in range(int(TIME_STEPS*2/3)):
        neuron.time_step(current, i)
    for i in range(TIME_STEPS-int(TIME_STEPS*2/3)):
        neuron.time_step(-1*current, i)
    neuron.add_plot(plt, 'r')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, SPIKE_VOLTAGE*SPIKE_MULTIPLIER))
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.grid(True)
    plt.minorticks_on()
    plt.gca().set_yticks([REST_VOLTAGE, SPIKE_VOLTAGE])
    plt.gca().set_xticks([])
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.25, wspace=0.25)
plt.show()


print('Problem 2')
plt.figure(2)
TIME_STEPS = 100
X_VALS, Y_VALS = ([], [])
for current in range(1, 400):
    X_VALS.append(current)
    neuron = LIF(CAPACITANCE, RESISTANCE, SPIKE_VOLTAGE, REST_VOLTAGE)
    for j in range(TIME_STEPS):
        neuron.time_step(current)
    spike_frequency = neuron.wave.values.count(SPIKE_VOLTAGE*SPIKE_MULTIPLIER)
    Y_VALS.append(spike_frequency/TIME_STEPS)
plt.plot(X_VALS, Y_VALS)
plt.title('LIF Neuron: Input Current vs. Firing Rate')
plt.xlabel('Input Current')
plt.ylabel('Firing Rate')
plt.show()


print('Problem 3')
plt.figure(3)
TIME_STEPS = 400
IZHI_U = -65*0.2
IZHI_PARAMS = (0.02, 0.2, -65, 2) # a,b,c,d
CURRENTS = []
for x in range(0, 20, 2):
    CURRENTS.append(x)
for i in range(0, len(CURRENTS)):
    neuron = Izhikevich(IZHI_U, IZHI_PARAMS, TIME_STEPS)
    current = CURRENTS[i]
    plt.subplot(6, 2, i+1)
    plt.title('Current: {}'.format(current))
    for i in range(TIME_STEPS-1):
        neuron.time_step(current)
    neuron.add_plot(plt)
    plt.grid(True)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([IZHI_PARAMS[2], IZHIKEVICH_SPIKE])
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, IZHI_PARAMS[2], IZHIKEVICH_SPIKE*SPIKE_MULTIPLIER))
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.25, wspace=0.25)
plt.show()


print('Problem 4: TTX')
plt.figure(4)
CURRENTS = []
INITIAL_CONDITIONS = {'C_m': 1.0, 'V': -65,\
        'h': 0.6, 'm': 0.05, 'n': 0.32,\
        'g_Na': 120, 'g_K': 36, 'g_L': 0.3,\
        'E_Na': 50, 'E_K': -77, 'E_L': -54.387}
for x in range(3):
    i = x
    current = lambda t: 25*i
    CURRENTS.append(current)
CURRENTS.append(lambda t: 10*(t>100) - 10*(t>200) + 35*(t>300))
ODD_STRING = 'Current: 10*(t>100) - 10*(t>200) + 35*(t>300)'

for i in range(0, len(CURRENTS)):
    neuron = HodgkinHuxley(INITIAL_CONDITIONS)
    current = CURRENTS[i]
    for j in range(4):
        plt.subplot(4, 4, i*4 + j+1)
        ratio = 0.25*(j+1)
        if i == len(CURRENTS)-1:
            plt.title('Current: 10*(t>100) - 10*(t>200) + 35*(t>300), TTX Ratio: {}'.format(ratio))
        else:
            plt.title('Current: {}, TTX Ratio: {}'.format(current(0), ratio))
        neuron.all_time_steps(current, ratio, 0) # ttx_ratio, pronase_ratio
        neuron.add_plot(plt)
        plt.gca().set_xticks([])
        plt.grid(True)
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.25, wspace=0.25)
plt.show()

print('Problem 4: Pronase')
plt.figure(4)
CURRENTS = []
INITIAL_CONDITIONS = {'C_m': 1.0, 'V': -65,\
        'h': 0.6, 'm': 0.05, 'n': 0.32,\
        'g_Na': 120, 'g_K': 36, 'g_L': 0.3,\
        'E_Na': 50, 'E_K': -77, 'E_L': -54.387}
for x in range(3):
    i = x
    current = lambda t: 25*i
    CURRENTS.append(current)
CURRENTS.append(lambda t: 10*(t>100) - 10*(t>200) + 35*(t>300))
ODD_STRING = 'Current: 10*(t>100) - 10*(t>200) + 35*(t>300)'

for i in range(0, len(CURRENTS)):
    neuron = HodgkinHuxley(INITIAL_CONDITIONS)
    current = CURRENTS[i]
    for j in range(4):
        plt.subplot(4, 4, i*4 + j+1)
        ratio = 0.25*(j+1)
        if i == len(CURRENTS)-1:
            plt.title('Current: 10*(t>100) - 10*(t>200) + 35*(t>300), Pronase Ratio: {}'.format(ratio))
        else:
            plt.title('Current: {}, Pronase Ratio: {}'.format(current(0), ratio))
        neuron.all_time_steps(current, 0, ratio) # ttx_ratio, pronase_ratio
        neuron.add_plot(plt)
        plt.gca().set_xticks([])
        plt.grid(True)
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.25, wspace=0.25)
plt.show()
