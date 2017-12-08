""" Spiking Neural Network implementation """
import math
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from pprint import pprint as pp
from collections import defaultdict
from itertools import tee, izip, product
from sklearn.preprocessing import normalize
from neurons import LIF as Neuron

VERBOSE = False

def pairwise(iterable):
    a, b = tee(iterable)
    b.next()
    return izip(a, b)

class Synapse(object):
    def __init__(self, presynaptic, postsynaptic):
        self.pre = presynaptic
        self.post = postsynaptic
        self.pre_trace = 0  # x_j
        self.post_trace = 0 # y_i
        self.weight = uniform(0, 1)

    def a(self, sign):
        if sign == '+': return self.weight/4.
        if sign == '-': return -self.weight/4.

    def tau(self, sign): # > 1
        if sign == '+': return 4.
        if sign == '-': return 4.

    def dirac(self, neuron, c_time):
        return neuron.last_spike == c_time

    def update(self, c_time):
        s = self
        s.pre_trace  += -s.pre_trace/s.tau('+')  + s.dirac(self.pre, c_time)
        s.post_trace += -s.post_trace/s.tau('-') + s.dirac(self.post, c_time)

        s.weight += s.a('-') * self.post_trace * s.dirac(self.pre, c_time) + \
                    s.a('+') * self.pre_trace  * s.dirac(self.post, c_time)


class SNN(object):
    def __init__(self, n_layers):
        s = self
        s.time_const = 3.
        s.learn_rate = 7.
        s.excit_ratio = 2
        s.trained = defaultdict()
        s.layers = [[Neuron(i_layer=i_layer, i_neuron=i_neuron)
                    for i_neuron in range(n_layers[i_layer])]
                        for i_layer in range(len(n_layers))]
        s.synapses = [s.gen_synapse_layer(pre_layer, post_layer)
                      for pre_layer, post_layer in pairwise(s.layers)]
        # make some of hidden layer inhibitory
        #hid_out = s.synapses[len(s.layers)/2]
        for layer in s.layers[0:int(math.ceil(len(s.layers)/s.excit_ratio))]:
            for neuron in layer: neuron.type = 'inhibitory'
            #for j in range(len(hid_out[i])): hid_out[i][j] *= -1

    def gen_synapse_layer(self, pre_layer, post_layer):
        synapses = defaultdict()
        for _pre in pre_layer:
            synapses[_pre] = defaultdict()
            for _post in post_layer:
                synapses[_pre][_post] = Synapse(_pre, _post)
        return synapses

    def train(self, problem, inp_maps, out_maps):
        if not problem in self.trained:
            if problem == 'XOR':
                self.reset()
                self.train_xor(inp_maps, out_maps)
                self.reset()
            self.trained[problem] = True

    def train_xor(self, inp_maps, out_maps):
        n_runs = 4
        all_inputs = map(lambda ttfs_pair: [self.calibrate_current(ttfs)
                for ttfs in ttfs_pair], list(product(inp_maps, inp_maps)))

        for input_currs in all_inputs:
            for i_run in range(1, n_runs): # train this many times
                self.reset() # is this needed?
                target_ttfs = out_maps[0] if input_currs[0] == input_currs[1] else out_maps[1]
                for time in range(1, (1+target_ttfs)):#*2):

                    # update all neurons
                    for layer in self.layers:
                        for neuron in layer:
                            current = input_currs[neuron.i_neuron] \
                                if neuron.i_layer == 0 else self.sum_epsps(time, neuron)
                            if neuron.i_layer == 2: # teaching input
                                current += 1000 if time % target_ttfs == 0 else -current
                            #if neuron.i_layer == 0 and time > target_ttfs: current = 0
                            neuron.time_step(current, time)

                    # update all weights
                    for synapse_layer in self.synapses:
                        for _pre, _post_layer in synapse_layer.items():
                            for _post, synapse in _post_layer.items():
                                synapse.update(time)

                self.normalize_weights()
                self.output_weights()

                for inp in list(product([0,1], [0,1])): # all input pairs
                    print('{}: {}'.format(inp, snn.classify(name, inp, inp_maps, out_maps)))
                print('')

        print('Trained!')

    def normalize_weights(self):
         col_sums = defaultdict(float)
         for synapse_layer in snn.synapses:
             for _pre, _post_synapses in synapse_layer.items():
                 for _post, synapse in _post_synapses.items():
                     col_sums[_post] += synapse.weight
         for synapse_layer in snn.synapses:
             for _pre, _post_synapses in synapse_layer.items():
                 for _post, synapse in _post_synapses.items():
                     col_sums[_post] += synapse.weight

    def output_weights(self):
        for synapse_layer in snn.synapses:
            for _pre, _post_synapses in synapse_layer.items():
                synapses = list(_post_synapses.values())
                weights = list(map(lambda syn: syn.weight, synapses))
                print('\t{}'.format(weights))

    def classify(self, problem, inputs, inp_maps, out_maps):
        self.reset()
        if problem == 'XOR':
            return self.classify_xor(inputs, inp_maps, out_maps)
        self.reset()

    def classify_xor(self, inputs, inp_maps, out_maps):
        output = self.layers[-1][0]
        inp_to_curr = {i : self.calibrate_current(inp_maps[i])
                       for i in range(len(inputs))}
        for time in range(1, 1000):
            for layer in self.layers:
                for neuron in layer:
                    current = inp_to_curr[inputs[neuron.i_neuron]] \
                        if neuron.i_layer == 0 else self.sum_epsps(time, neuron)
                    neuron.time_step(current, time)
            if output.has_spiked(): return output.last_spike

    def sum_epsps(self, time, neuron):
        current = 0
        prev = neuron.i_layer-1
        for _pre in self.layers[prev]:
            if _pre.has_spiked():
                ls, wt = _pre.last_spike, self.synapses[prev][_pre][neuron].weight
                if neuron.i_layer == 2 and _pre.type == 'inhibitory': wt *= -1
                current += wt * Neuron().v_max * math.e**(-(time-ls)/self.time_const)
        return current

    def calibrate_current(self, ttfs):
        for current in range(0, 10000, 1):
            neuron = Neuron()
            for time in range(0, ttfs+1):
                neuron.time_step(current/10, time)
                if neuron.last_spike != -1: break
            if neuron.last_spike == ttfs:
                return current/10.0

    def reset(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.last_spike = -1


if __name__ == '__main__':

    snn = SNN([2, 10, 1])
    problems = defaultdict()
    problems['XOR'] = [[0, 1], [6, 3], [11, 8]]
    #problems['XOR'] = [[0, 1], [4, 1], [10, 15]]

    for name, vals in problems.items():

        inps, inp_maps, out_maps = vals
        print('\n{}\n{}'.format(name, '-'*len(name)))

        print('Training...')
        snn.train(name, inp_maps, out_maps)
        snn.output_weights()

        print('Classifying...')
        for inp in list(product(inps, inps)): # all input pairs
            print('{}: {}'.format(inp, snn.classify(name, inp, inp_maps, out_maps)))

        #snn.layers[-1][0].add_plot(plt)
        #plt.show()
