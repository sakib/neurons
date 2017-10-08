""" Neuron Simulator """
#from neurons import LIF, Izhikevich, HodgkinHuxley
from neurons import HodgkinHuxley
from neurons import TIME_STEPS#, IZHIKEVICH_SPIKE, SPIKE_MULTIPLIER
import matplotlib.pyplot as plt
#from matplotlib.ticker import NullFormatter

CURRENTS = []
RESISTANCE = 2
CAPACITANCE = 2.5
REST_VOLTAGE = 10
SPIKE_VOLTAGE = 75

"""
# Problem 1
plt.figure(1)
for x in range(10, 90, 20):
    CURRENTS.append(x)
for i in range(1, len(CURRENTS)+1):
    neuron = LIF(CAPACITANCE, RESISTANCE, SPIKE_VOLTAGE, REST_VOLTAGE)
    current = CURRENTS[i-1]
    plt.subplot(810+i)
    plt.title('Current: {}'.format(current))
    for i in range(TIME_STEPS-1):
        neuron.time_step(current)
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
        neuron.time_step(current)
    for i in range(TIME_STEPS-int(TIME_STEPS*2/3)):
        neuron.time_step(0)
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

# Problem 2
plt.figure(2)
TIME_STEPS = 100
X_VALS, Y_VALS = ([], [])
for current in range(1, 100000):
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

# Problem 3
plt.figure(3)
IZHI_U = -65*0.2
IZHI_PARAMS = (0.02, 0.2, -65, 2) # a,b,c,d
CURRENTS = []
for x in range(0, 20, 2):
    CURRENTS.append(x)
for i in range(0, len(CURRENTS)):
    neuron = Izhikevich(IZHI_U, IZHI_PARAMS)
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
"""
# Problem 4
plt.figure(4)
HH_G = {'Na': 120, 'K': 36, 'L': 0.3}
HH_E = {'Na': 50, 'K': -77, 'L': -54.387}
HH_OPTS = {'h': 0.6, 'm': 0.05, 'n': 0.32}
HH_CAPACITANCE = 1.0
INITIAL_VOLTAGE = -65
CURRENTS = []
for x in range(0, 100, 10):
    CURRENTS.append(x)
for i in range(0, len(CURRENTS)):
    neuron = HodgkinHuxley(HH_CAPACITANCE, HH_G, HH_E, HH_OPTS, INITIAL_VOLTAGE)
    current = CURRENTS[i]
    plt.subplot(5, 2, i+1)
    plt.title('Current: {}'.format(current))
    for i in range(TIME_STEPS-1):
        neuron.time_step(current)
    neuron.add_plot(plt)
    plt.grid(True)
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.25, wspace=0.25)
plt.show()
