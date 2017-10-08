import abc
import operator
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_WAVELENGTH = 100

class Wave:
    """ Moving set of values """
    def __init__(self, cardinality):
        self.values = []
        assert cardinality > 0
        self.cardinality = cardinality
        self.time_axis = [i for i in range(len(DEFAULT_WAVELENGTH))]

    def append(self, new_value):
        if len(self.values) == cardinality:
            self.values.pop(0)
        self.values.append(new_value)

    def display(self):
        x_values = np.arange(self.time_axis)
        y_values = np.arange(self.values)
        plt.plot(x_values, y_values)
        plt.show()

class Neuron(object):
    """ Abstract base class for Neurons """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        self.wave = Wave(DEFAULT_WAVELENGTH)
        pass

    @abc.abstractmethod
    def time_step(self, current):
        pass

class LIF(Neuron):
    """ Implementation of leaky integrate-and-fire neuron model """
    def __init__(self, capacitance, resistance, spike_voltage, rest_voltage):
        self.voltage = 0
        self.res = resistance
        self.cap = capacitance
        self.v_max = spike_voltage
        self.v_rest = rest_voltage

    def time_step(self, current):
        self.voltage += current/self.cap - self.voltage/(self.cap*self.res)
        if self.voltage > self.v_max:
            self.voltage = self.v_rest
            self.wave.append(self.v_max*2)
            self.wave.append(0)
        elif self.voltage < self.v_rest:
            self.voltage = self.v_rest
        self.wave.append(self.voltage)

