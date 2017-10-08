""" Neuron implementation """
import abc
from math import exp
import scipy as sp

TIME_STEPS = 250
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
        self.time_axis = [i for i in range(TIME_STEPS)]

    def append(self, new_value):
        """ Add value to end of the series """
        if len(self.values) == self.cardinality:
            self.values.pop(0)
        self.values.append(new_value)

    def add_plot(self, plt, color=None):
        """ Display value series as curve """
        x_values = sp.array(self.time_axis)
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
    def __init__(self, hh_cap, hh_g, hh_E, hh_opts, init_v):
        Neuron.__init__(self)
        self.voltage = init_v
        self.C_m = hh_cap
        self.p_g = hh_g
        self.p_e = hh_E
        self.opts = hh_opts
        self.wave.append(init_v)

    def time_step(self, current):
        a = self.get_alphas()
        b = self.get_betas()
        I = self.get_currents()

        h, m, n = self.opts['h'], self.opts['m'], self.opts['n']
        self.voltage += (current - I['Na'] - I['K'] - I['L']) / self.C_m
        self.opts['h'] += a['h']*(1-h) - b['h']*h
        self.opts['m'] += a['m']*(1-m) - b['m']*m
        self.opts['n'] += a['n']*(1-n) - b['n']*n

        self.wave.append(self.voltage)
        print(self.voltage)

    def get_alphas(self):
        """ Get alpha parameters """
        alphas = {}
        V = self.voltage
        alphas['h'] = 0.07*exp(-(V+65)/20)
        alphas['m'] = 0.1*(V+40)/(1 - exp(-(V+40)/10))
        alphas['n'] = 0.01*(V+55)/(1 - exp(-(V+55)/10))
        return alphas

    def get_betas(self):
        """ Get beta parameters """
        betas = {}
        V = self.voltage
        betas['h'] = 1/(1 + exp(-(V+35)/10))
        betas['m'] = 4*exp(-(V+65)/18)
        betas['n'] = 0.125*exp(-(V+65)/80)
        return betas

    def get_currents(self):
        """ Get membrane currents """
        currents = {}
        V = self.voltage
        h, m, n = self.opts['h'], self.opts['m'], self.opts['n']
        g_Na, g_K, g_L = self.p_g['Na'], self.p_g['K'], self.p_g['L']
        E_Na, E_K, E_L = self.p_e['Na'], self.p_e['K'], self.p_e['L']
        currents['Na'] = g_Na * m**3 * h * (V - E_Na)
        currents['K'] =  g_K  * n**4     * (V - E_K)
        currents['L'] =  g_L             * (V - E_L)
        print(h, m, n)
        print(currents)
        return currents

    def add_plot(self, plt, color=None):
        self.wave.add_plot(plt, color)
