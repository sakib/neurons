""" Neuron implementation """
import abc
import numpy as np

TIME_STEPS = 250
SPIKE_MULTIPLIER = 2.5
IZHIKEVICH_SPIKE = 30 # 30 mV

class Wave:
    """ Moving set of values """
    def __init__(self, cardinality):
        self.values = []
        assert cardinality > 0
        self.cardinality = cardinality
        self.time_axis = [i for i in range(TIME_STEPS)]

    def append(self, new_value):
        """ Add value to end of the series """
        if len(self.values) == self.cardinality:
            self.values.pop(0)
        self.values.append(new_value)

    def add_plot(self, plt, color=None):
        """ Display value series as curve """
        x_values = np.array(self.time_axis)
        y_values = np.array(self.values)
        if color is None:
            plt.plot(x_values, y_values)
        else:
            plt.plot(x_values, y_values, color)

class Neuron(object):
    """ Abstract base class for Neurons """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        self.wave = Wave(TIME_STEPS)

    @abc.abstractmethod
    def time_step(self, current):
        """ Update voltage given a current at a time """
        pass

    @abc.abstractmethod
    def add_plot(self, plt, color=None):
        """ Adds neuron spike train to given pyplot """
        pass

class LIF(Neuron):
    """ Implementation of leaky integrate-and-fire neuron model """
    def __init__(self, capacitance, resistance, spike_voltage, rest_voltage):
        Neuron.__init__(self)
        self.voltage = rest_voltage
        self.res = resistance
        self.cap = capacitance
        self.v_max = spike_voltage
        self.v_rest = rest_voltage
        self.wave.append(rest_voltage)

    def time_step(self, current):
        self.voltage += current/self.cap
        self.voltage -= self.voltage/(self.cap*self.res)
        if self.voltage > self.v_max:
            self.voltage = self.v_rest
            self.wave.append(self.v_max*SPIKE_MULTIPLIER)
            self.wave.append(0)
        elif self.voltage < self.v_rest:
            self.voltage = self.v_rest
        self.wave.append(self.voltage)

    def add_plot(self, plt, color=None):
        self.wave.add_plot(plt, color)

class Izhikevich(Neuron):
    """ Implementation of Izhikevich neuron model """
    def __init__(self, p_u, params):
        Neuron.__init__(self)
        p_a, p_b, p_c, p_d = params
        self.voltage = p_c
        self.p_u = p_u
        self.p_a = p_a
        self.p_b = p_b
        self.p_c = p_c
        self.p_d = p_d
        self.wave.append(p_c)

    def time_step(self, current):
        self.voltage += (0.04*(self.voltage**2) + 5*self.voltage + 140 - self.p_u + current)
        self.p_u += (self.p_a * (self.p_b*self.voltage - self.p_u))
        if self.voltage >= IZHIKEVICH_SPIKE:
            self.voltage = self.p_c
            self.p_u += self.p_d
            self.wave.append(IZHIKEVICH_SPIKE*SPIKE_MULTIPLIER)
        elif self.voltage < self.p_c:
            self.voltage = self.p_c
        self.wave.append(self.voltage)

    def add_plot(self, plt, color=None):
        self.wave.add_plot(plt, color)

class HodgkinHuxley(Neuron):
    """ Implementation of Hodgkin-Huxley neuron model """
    def __init__(self, hh_capacitance, hh_opts, hh_g, hh_v):
        Neuron.__init__(self)
        self.voltage = 0 #?
        self.cap = hh_capacitance
        self.opts = hh_opts
        self.p_g = hh_g
        self.p_v = hh_v
        self.wave.append(0) #?

    def time_step(self, current):
        v_diffs = {}
        v_diffs['Na'] = self.p_g['Na']*\
                (self.opts['m']**3)*self.opts['h']*(self.voltage-self.p_v['Na'])
        v_diffs['K'] = self.p_g['K']*\
                (self.opts['n']**4)*(self.voltage-self.p_v['K'])
        v_diffs['L'] = self.p_g['L']*\
                (self.voltage-self.p_v['L'])
        v_diff = (current - v_diffs['Na'] - v_diffs['K'] - v_diffs['L'])/self.cap
        self.voltage += v_diff
        if self.voltage >= HH_SPIKE:
            # spike
        elif self.voltage < self.minimum: #?
            # reset
        self.wave.append(self.voltage)

    def add_plot(self, plt, color=None):
        self.wave.add_plot(plt, color)
