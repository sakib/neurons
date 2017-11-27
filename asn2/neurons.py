""" Neuron implementation """
import abc
from math import exp
import scipy as sp
from scipy.integrate import odeint

TIME_STEPS = 400
SPIKE_MULTIPLIER = 2.5
IZHIKEVICH_SPIKE = 30 # 30 mV
HH_SPIKE = 30 # 30 mV
HH_VOLT_MIN = -60

class Wave:
    """ Moving set of values """
    def __init__(self, cardinality):
        self.values = []
        assert cardinality > 0
        self.cardinality = cardinality
        self.time_axis = sp.array([i for i in range(TIME_STEPS)])

    def append(self, new_value):
        """ Add value to end of the series """
        if len(self.values) == self.cardinality:
            self.values.pop(0)
        self.values.append(new_value)

    def add_plot(self, plt, color=None):
        """ Display value series as curve """
        x_values = sp.array([i for i in range(len(self.values))])
        y_values = sp.array(self.values)
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
        self.last_spike = -1

    @abc.abstractmethod
    def time_step(self, current):
        """ Update voltage given a current at a time """
        pass

    @abc.abstractmethod
    def reset_spike(self):
        pass

    @abc.abstractmethod
    def add_plot(self, plt, color=None):
        """ Adds neuron spike train to given pyplot """
        pass

class LIF(Neuron):
    """ Implementation of leaky integrate-and-fire neuron model """
    #def __init__(self, capacitance=2.5, resistance=2, spike_voltage=75., rest_voltage=0., type='excitatory', i_layer=-1, i_neuron=-1):
    def __init__(self, capacitance=2.5, resistance=2, spike_voltage=20., rest_voltage=5., type='excitatory', i_layer=-1, i_neuron=-1):
        Neuron.__init__(self)
        self.spikes = []
        self.i_layer = i_layer
        self.i_neuron = i_neuron
        self.type = type
        self.voltage = rest_voltage
        self.res = resistance
        self.cap = capacitance
        self.v_max = spike_voltage
        self.v_rest = rest_voltage
        self.last_spike = -1
        self.wave.append(rest_voltage)

    def time_step(self, current, curr_time):
        self.voltage += current/self.cap
        self.voltage -= self.voltage/(self.cap*self.res)
        self.wave.append(min(self.voltage, self.v_max))
        if self.voltage > self.v_max: # spike
            self.spikes.append(curr_time)
            self.last_spike = curr_time
            self.voltage = self.v_rest
            self.wave.append(self.v_max*SPIKE_MULTIPLIER)
            self.wave.append(0)
            self.wave.append(self.voltage)
            return 1
        elif self.voltage < self.v_rest:
            self.voltage = self.v_rest
            self.wave.append(self.voltage)
        return 0

    def get_last_spike(self):
        return self.last_spike

    def reset_spike(self):
        self.last_spike = -1

    def reset(self):
        self.last_spike = -1
        self.spikes = []

    def has_spiked(self):
        return not self.last_spike == -1

    def just_spiked(self, time):
        return self.last_spike == time

    def add_plot(self, plt, color=None):
        self.wave.add_plot(plt, color)

class Izhikevich(Neuron):
    """ Implementation of Izhikevich neuron model """
    def __init__(self, p_u, params, time_steps):
        Neuron.__init__(self)
        p_a, p_b, p_c, p_d = params
        self.voltage = p_c
        self.p_u = p_u
        self.p_a = p_a
        self.p_b = p_b
        self.p_c = p_c
        self.p_d = p_d
        self.wave = Wave(time_steps)
        self.wave.append(p_c)

    def time_step(self, current=0):
        self.voltage += (0.04*(self.voltage**2) + 5*self.voltage + 140 - self.p_u + current)
        self.p_u += (self.p_a * (self.p_b*self.voltage - self.p_u))
        if self.voltage >= IZHIKEVICH_SPIKE:
            self.voltage = self.p_c
            self.p_u += self.p_d
            self.wave.append(IZHIKEVICH_SPIKE*SPIKE_MULTIPLIER)
        elif self.voltage < self.p_c:
            self.voltage = self.p_c
        self.wave.append(self.voltage)

    def reset_spike(self):
        self.last_spike = -1

    def add_plot(self, plt, color=None):
        self.wave.add_plot(plt, color)

class HodgkinHuxley(Neuron):
    """ Implementation of Hodgkin-Huxley neuron model """
    def __init__(self, IC):
        Neuron.__init__(self)
        self.IC = IC # initial conditions
        self.a, self.b, self.I = {}, {}, {} # alphas, betas, currents
        self.a['h'] = lambda V: 0.07*exp(-(V+65)/20)
        self.a['m'] = lambda V: 0.1*(V+40)/(1 - exp(-(V+40)/10))
        self.a['n'] = lambda V: 0.01*(V+55)/(1 - exp(-(V+55)/10))
        self.b['h'] = lambda V: 1/(1 + exp(-(V+35)/10))
        self.b['m'] = lambda V: 4*exp(-(V+65)/18)
        self.b['n'] = lambda V: 0.125*exp(-(V+65)/80)
        self.I['Na'] = lambda V, m, h, p: IC['g_Na'] * m**3 * (1-p)*h * (V - IC['E_Na'])
        self.I['K'] = lambda V, n:     IC['g_K']  * n**4     * (V - IC['E_K'])
        self.I['L'] = lambda V:        IC['g_L']             * (V - IC['E_L'])

    def all_time_steps(self, current, ttx_ratio=0, pronase_ratio=0):
        a, b, I, IC = self.a, self.b, self.I, self.IC
        def ODEs(variables, times):
            V, h, m, n = variables
            dVdt = (current(times) - (1-ttx_ratio)*(I['Na'](V, m, h, pronase_ratio)) - I['K'](V, n) - I['L'](V)) / IC['C_m']
            dhdt = a['h'](V)*(1-h) - b['h'](V)*h
            dmdt = a['m'](V)*(1-m) - b['m'](V)*m
            dndt = a['n'](V)*(1-n) - b['n'](V)*n
            return dVdt, dhdt, dmdt, dndt

        X = odeint(ODEs, [IC['V'], IC['h'], IC['m'], IC['n']], self.wave.time_axis)
        for voltage in X[:,0]:
            self.wave.append(voltage)

    def reset_spike(self):
        self.last_spike = -1

    def add_plot(self, plt, color=None):
        self.wave.add_plot(plt, color)
