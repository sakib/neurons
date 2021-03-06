""" Tempotron Implementation """
import sys, math, time, random
from collections import defaultdict
from neurons import LIF as Neuron
import matplotlib.pyplot as plt

SMOL = 1. #2.5
VALIDS = ['*']


def round(flt, n_decimals=1):
    return math.ceil(flt*10**n_decimals)/10**n_decimals


def image_to_windows(image, n_neurons):
    with open('lines/{}'.format(image), 'r') as img:
        chars = [list(line.strip()) for line in img]

    windows = []
    l_win = math.sqrt(n_neurons)
    n_windows = 1 + len(chars) - int(l_win)

    if len(chars) != len(chars[0]): sys.exit('image must be square')
    if l_win - int(l_win) != 0: sys.exit('perception must be square')
    if n_windows <= 0: sys.exit('image must be larger than perception')

    for r in range(n_windows):
        for c in range(n_windows):
            windows.append([row[c:c+int(l_win)] for row in chars[r:r+int(l_win)]])

    return windows


def matrix_to_windows(matrix, n_neurons):
    windows = []
    l_win = 3
    n_windows = 8

    if len(matrix) != len(matrix[0]): sys.exit('image must be square')
    if l_win - int(l_win) != 0: sys.exit('perception must be square')
    if n_windows <= 0: sys.exit('image must be larger than perception')

    for r in range(n_windows):
        for c in range(n_windows):
            windows.append([row[c:c+int(l_win)] for row in matrix[r:r+int(l_win)]])

    return windows


class Tempotron(object):
    def __init__(self, n_inputs):
        s = self
        s.lambduh = .25     # learning rate
        s.on_ttfs = 3       # ms when 'on' neurons should spike
        s.t_threshold = 10  # time to wait for output to spike
        s.V_rest, s.V_th = 0, n_inputs
        s.neurons = [Neuron(i_neuron=i) for i in range(n_inputs)]
        
        s.synapses = [random.uniform(-1, 1) for neuron in s.neurons]
        for i in range(len(s.neurons)):
            if i >= math.sqrt(n_inputs):
                s.synapses[i] = random.uniform(0, 1) # limit inhibs


    def tau(self, sign='none'): # > 1
        if sign == 's': return SMOL
        else: return 4 * SMOL


    def weight_delta(self, neuron, t_max):
        if not neuron.has_spiked() or neuron.just_spiked(t_max): return 0
        psps = self.psp_kernels(neuron, t_max)
        if len(psps) == 0: return 0
        else: return self.lambduh * sum(psps) * 1./max(psps)


    def train(self, image, switch_fn, trueImage):
        s = self
        s.reset()
        classified, t_max, v_max = s.classify(image, switch_fn)
        weight_deltas = [s.weight_delta(neuron, t_max) for neuron in s.neurons]
        if not classified and trueImage: # increase all weights
            s.synapses = [w+dw for w,dw in zip(s.synapses, weight_deltas)]
        elif classified and not trueImage: # decrease all weights
            s.synapses = [w-dw for w,dw in zip(s.synapses, weight_deltas)]


    def classify(self, image, switch_fn, plot=None):
        s = self
        s.reset()
        t_max, v_max = 0, 0
        x_axis, y_axis = [], []

        # process input image
        windows = matrix_to_windows(image, len(s.neurons))
        switches = list(map(switch_fn, windows))
        for time in range(s.t_threshold):
            for idx in range(len(s.neurons)):
                if switches[idx]: # make neuron spike
                    current = s.calibrate_current(s.on_ttfs)
                    s.neurons[idx].time_step(current, time)
            # calculate output voltage V(t) and track time of max voltage
            c_voltage = s.voltage(time)
            x_axis.append(time)
            y_axis.append(min(c_voltage, s.V_th))
            if c_voltage > v_max:
                t_max = time
                v_max = c_voltage
            if c_voltage >= s.V_th: # >
                if plot: plot.plot(x_axis, y_axis)
                return True, t_max, v_max

        if plot: plot.plot(x_axis, y_axis)
        return False, t_max, v_max


    def voltage(self, time):
        s = self
        voltage = s.V_rest
        for neuron in s.neurons:
            weight = s.synapses[neuron.i_neuron]
            if neuron.has_spiked() and not neuron.just_spiked(time):
                psps = s.psp_kernels(neuron, time)
                voltage += weight * sum(psps) * 1./max(psps)
        return voltage


    def psp_kernels(self, neuron, time):
        psps = []
        for t_spike in neuron.spikes:
            if time > t_spike:
                kernel = math.e**(-(time-t_spike)/self.tau()) - \
                         math.e**(-(time-t_spike)/self.tau('s'))
                psps.append(kernel)
        return psps


    def calibrate_current(self, ttfs):
        for current in range(0, 10000, 1):
            neuron = Neuron()
            for time in range(0, ttfs+1):
                neuron.time_step(current/10., time)
                if neuron.has_spiked(): break
            if neuron.last_spike == ttfs:
                return current/10.


    def reset(self):
        for neuron in self.neurons:
            neuron.reset()


    def print_synapses(self):
        sqrt = int(math.sqrt(len(self.synapses)))
        for i in range(0, len(self.synapses), sqrt):
            print(self.synapses[i:i+sqrt])


    def window_to_valid(self, window):
        #window is 2d array
        for i in range(len(window)):
            for j in window[i]:
                sum += j

        avg = sum/(len(window) * len(window[0]))

        return avg >= 128