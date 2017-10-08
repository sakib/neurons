# neuron simulator
from neurons import LIF

currents = []

def constant_current(t):
    return 50

currents.append(constant_current)

# cap, res, spike, rest
neuron = LIF(10, 10, 50, 10)

for i in range(100):
    neuron.time_step(currents[0](i))
    print neuron.voltage

neuron.wave.display()
