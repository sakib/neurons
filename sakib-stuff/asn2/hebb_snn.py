""" Spiking Neural Network implementation """
import math
import numpy as np
from pprint import pprint as pp
from collections import defaultdict
from itertools import tee, izip, product
from sklearn.preprocessing import normalize
from neurons import LIF as Neuron

VERBOSE = False

def pairwise(iterable):
    a, b = tee(iterable)
    next(b)
    return zip(a, b)


class SNN(object):
    def __init__(self, n_layers):
        self.time_const = 3.
        self.learn_rate = 7.
        self.excit_ratio = 10
        self.t_window = 50
        self.trained = defaultdict()
        self.layers = [[Neuron() for i in range(n)] for n in n_layers]
        self.synapses = [np.random.rand(len1, len2) for len1, len2 in pairwise(n_layers)]
        # make half of hidden layer inhibitory
        #mid, hid_out = list(map(lambda l: l[len(self.layers)/2], [self.layers, self.synapses]))
        #for i in range(int(math.ceil(len(mid)/self.excit_ratio)), len(mid)):
        #    mid[i].type = 'inhibitory' # invert inhib weights
        #    for j in range(len(hid_out[i])): hid_out[i][j] *= -1


    def train(self, problem, inp_maps, out_maps):
        if not problem in self.trained:
            if problem == 'XOR':
                self.reset()
                self.train_xor(inp_maps, out_maps)
                self.reset()
            self.trained[problem] = True


    def train_xor(self, inp_maps, out_maps):

        n_spikes, n_windows = 0, 1#200
        t_threshold = self.t_window * n_windows
        output = self.layers[-1][0] # output neuron
        all_inputs = map(lambda rate_pair: [self.calibrate_current(rate)
                for rate in rate_pair], list(product(inp_maps, inp_maps)))

        for input_currs in all_inputs:
            target_rate = out_maps[0] if input_currs[0] == input_currs[1] else out_maps[1]
            print('\ntraining {} to yield output rate {}...'.format(input_currs, target_rate))

            for i_window in range(n_windows): #0-200
                self.reset()
                spikes = defaultdict(int) # number of spikes in the past window
                for c_time in range(1, 1+self.t_window): #0-50
                    #print('the time is {}ms'.format(c_time))
                    for i_c_layer in range(len(self.layers)):
                        for i_c_neuron in range(len(self.layers[i_c_layer])):
                            neuron = self.layers[i_c_layer][i_c_neuron]
                            if i_c_layer == 0: curr = input_currs[i_c_neuron]
                            else: curr = self.sum_epsps(c_time, i_c_layer, i_c_neuron)
                            if neuron is output:
                                if c_time % (self.t_window/target_rate) == 0: curr += 1000
                                else: curr = 0
                            if neuron.time_step(curr, c_time) == 1: spikes[neuron] += 1
                            #elif neuron.last_spike == c_time: print('\tlayer {} neuron {}
                                #type {} spikes!'.format(i_c_layer, i_c_neuron, neuron.type))
                #print('{}'.format(spikes[output]))
                self.update_weights(c_time, spikes)
                #print('the weights are being updated!')
                #for synapse in self.synapses:
                #    for row in synapse: print('\t{}'.format(row))
            for synapse in snn.synapses:
                for row in synapse: print('\t{}'.format(row))
            print('(0,0): {}'.format(self.classify_xor([0, 0], inp_maps, out_maps)))
            print('(0,1): {}'.format(self.classify_xor([0, 1], inp_maps, out_maps)))
            print('(1,0): {}'.format(self.classify_xor([1, 0], inp_maps, out_maps)))
            print('(1,1): {}'.format(self.classify_xor([1, 1], inp_maps, out_maps)))

        print('Trained!')


    def update_weights(self, c_time, spikes):

        for i_c_synapse in range(len(self.synapses)):
            c_synapse = self.synapses[i_c_synapse]
            pre_layer, post_layer = self.layers[i_c_synapse:i_c_synapse+2]
            for _post in post_layer:
                for _pre in pre_layer:
                    r, c = pre_layer.index(_pre), post_layer.index(_post)
                    c_synapse[r][c] += (spikes[_pre]*spikes[_post] - \
                        c_synapse[r][c]*spikes[_pre]**2) * 1./self.learn_rate

            # normalize
            col_sums = defaultdict(int)
            for r in range(len(c_synapse)):
                for c in range(len(c_synapse[r])):
                    col_sums[c] += c_synapse[r][c]
            for r in range(len(c_synapse)):
                for c in range(len(c_synapse[r])):
                    c_synapse[r][c] /= col_sums[c]


    def classify(self, problem, inputs, inp_maps, out_maps):
        self.reset()
        if problem == 'XOR':
            return self.classify_xor(inputs, inp_maps, out_maps)
        self.reset()


    def classify_xor(self, inputs, inp_maps, out_maps):
        inp_to_curr = {i : self.calibrate_current(inp_maps[i])
                       for i in range(len(inputs))}

        n_spikes, n_windows = 0, 20
        t_threshold = self.t_window * n_windows

        for c_time in range(1, t_threshold):
            for i_c_layer in range(len(self.layers)):
                for i_c_neuron in range(len(self.layers[i_c_layer])):
                    neuron = self.layers[i_c_layer][i_c_neuron]
                    if i_c_layer == 0: curr = inp_to_curr[inputs[i_c_neuron]]
                    else: curr = self.sum_epsps(c_time, i_c_layer, i_c_neuron)
                    spike = neuron.time_step(curr, c_time)
                    if i_c_layer == 2: n_spikes += spike
                    #if neuron.last_spike == c_time:
                    #    print('\ttime {}: layer {} neuron {} type {} spikes!'
                    #        .format(c_time, i_c_layer, i_c_neuron, neuron.type))
                    #print('t: {}, l: {}, n: {}, sp: {}, c: {}, v: {}'.format(c_time,
                    #    i_c_layer, i_c_neuron, neuron.last_spike, curr, neuron.voltage))

        return n_spikes/n_windows

    def sum_epsps(self, c_time, i_c_layer, i_c_neuron):
        current = 0
        c_layer = self.layers[i_c_layer]
        p_layer = self.layers[i_c_layer-1]
        c_synapse = self.synapses[i_c_layer-1]
        for i_p_neuron in range(len(p_layer)):
            p_neuron = p_layer[i_p_neuron]
            p_spike = p_neuron.last_spike
            if p_spike != -1: # ttfs exists
                weight = c_synapse[i_p_neuron][i_c_neuron]
                change = weight * Neuron().v_max * math.e**(-(c_time-p_spike)/self.time_const)
                #if VERBOSE: print('\tadding epsp to current: {}'.format(change))
                current += change
        return current


    def calibrate_current(self, rate):
        """ Rate must be less than self.t_window """
        for current in range(0, 1000, 1):
            neuron = Neuron()
            spikes = 0
            for c_time in range(1, self.t_window+1):
                if neuron.time_step(current/10., c_time) == 1:
                    spikes += 1
            if spikes == rate: return current/10.
            if spikes == self.t_window: break
        return -1


    def reset(self):
        """ Reset time-to-first-spikes for every neuron in the network """
        for layer in self.layers:
            for neuron in layer:
                neuron.last_spike = -1


if __name__ == '__main__':
    snn = SNN([2, 10, 1])
    problems = defaultdict()
    problems['XOR'] = [[0, 1], [12, 16], [5, 8]]

    for name, vals in problems.items():
        inps, inp_maps, out_maps = vals
        snn.train(name, inp_maps, out_maps)
        for synapse in snn.synapses:
            for row in synapse: print('\t{}'.format(row))
        for inp in product(inps, inps):
            print('{}: {}'.format(inp, snn.classify(name, inp, inp_maps, out_maps)))

